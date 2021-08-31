# import libraries
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import numpy as np
import torch


from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm, trange
from transformers import (AdamW, get_linear_schedule_with_warmup, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from utils import TextDataset, load_and_cache_examples, set_seed, _rotate_checkpoints
from model import VAE_config, VAE_GPT2



logger = logging.getLogger(__name__)
MODEL_CLASSES = {'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)}

EPS = 1e-12

"""====================== CLASSES AND METHODS DEFINITIONS ======================"""
        
class loss_calculation():
    
    def __init__(self, args, vae_config):
        self.cont_capacity = args.cont_capacity
        self.disc_capacity = args.disc_capacity
        self.logging_steps = args.logging_steps
        self.device = args.device
        self.model = vae_config
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=args.ignore_index)
        self.global_step = None

        # variables for recording losses
        self.losses = {'loss': [],
                       'recon_loss': [],
                       'kl_loss': []}

        # Keep track of divergence values for each latent variable
        if self.model.is_continuous:
            self.losses['kl_loss_cont'] = []
            # For every dimension of continuous latent variables
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)] = []

        if self.model.is_discrete:
            self.losses['kl_loss_disc'] = []
            # For every discrete latent variable
            for i in range(len(self.model.latent_spec['disc'])):
                self.losses['kl_loss_disc_' + str(i)] = []          

    def _loss_function(self, data, recon_data, latent_dist, tokenizer, global_step):

            # Update global_step
            self.global_step = global_step

            # Reconstruction loss 
            # transform decoder_lm_logits from [batch_size, seq_length, vocab_size] => [batch_size * seq_length, vocab_size], target from [batch_size, sweq_length] => [batch_size * sweq_length]
            recon_loss = self.loss_fct(recon_data.view(-1, recon_data.size(-1)), data.contiguous().view(-1))  

            # Calculate KL divergences
            kl_cont_loss = 0  # Used to compute capacity loss (but not a loss in itself)
            kl_disc_loss = 0  # Used to compute capacity loss (but not a loss in itself)
            cont_capacity_loss = 0
            disc_capacity_loss = 0

            if self.model.is_continuous:
                # Calculate KL divergence
                mean, logvar = latent_dist['cont']
                kl_cont_loss = self._kl_normal_loss(mean, logvar)
                # Linearly increase capacity of continuous channels
                cont_min, cont_max, cont_num_iters, cont_gamma = \
                    self.cont_capacity
                # Increase continuous capacity without exceeding cont_max
                cont_cap_current = (cont_max - cont_min) * self.global_step / float(cont_num_iters) + cont_min
                cont_cap_current = min(cont_cap_current, cont_max)
                # Calculate continuous capacity loss
                cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)

            if self.model.is_discrete:
                # Calculate KL divergence
                kl_disc_loss = self._kl_multiple_discrete_loss(latent_dist['disc'])
                # Linearly increase capacity of discrete channels
                disc_min, disc_max, disc_num_iters, disc_gamma = \
                    self.disc_capacity
                # Increase discrete capacity without exceeding disc_max or theoretical
                # maximum (i.e. sum of log of dimension of each discrete variable)
                disc_cap_current = (disc_max - disc_min) * self.global_step / float(disc_num_iters) + disc_min
                disc_cap_current = min(disc_cap_current, disc_max)
                # Require float conversion here to not end up with numpy float
                disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in self.model.latent_spec['disc']])
                disc_cap_current = min(disc_cap_current, disc_theoretical_max)
                # Calculate discrete capacity loss
                disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)

            # Calculate total kl value to record it
            kl_loss = kl_cont_loss + kl_disc_loss

            # Calculate total loss
            total_loss = recon_loss + cont_capacity_loss + disc_capacity_loss

            # Record losses
            if self.global_step % self.logging_steps == 1:
                self.losses['recon_loss'].append(recon_loss.item())
                self.losses['kl_loss'].append(kl_loss.item())
                self.losses['loss'].append(total_loss.item())

            ## DEBUGGING!
            # Losses
            rounded_total_loss = np.round(total_loss.item(),3)
            rounded_recon_loss = np.round(recon_loss.item(),3)
            rounded_kl_cont_loss = np.round(kl_cont_loss.item(),3)
            rounded_kl_disc_loss = np.round(kl_disc_loss.item(),3)
            print("Losses: total_loss({}) - recon_loss({}) +  kl_cont_loss({}) +  kl_disc_loss({})".format(str(rounded_total_loss), str(rounded_recon_loss), str(rounded_kl_cont_loss), str(rounded_kl_disc_loss)))
            # Decode to text
            predictions = torch.nn.functional.softmax(recon_data, dim = -1)
            logger.info("decoder_label: " + str(tokenizer.decode(data[0].tolist(), clean_up_tokenization_spaces=True).replace("[PAD]", "")))
            prediction_text = tokenizer.decode(torch.argmax(predictions[0], dim=-1).tolist(), clean_up_tokenization_spaces=True)
            first_endoftext = prediction_text.find("<|endoftext|>") 
            logger.info("predictions: " + str(prediction_text[:(first_endoftext + 13) if first_endoftext>0 else len(prediction_text)])) 



            # To avoid large losses normalise by number of pixels
            return total_loss 

    def _kl_normal_loss(self, mean, logvar):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        # Calculate KL divergence
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        # Mean KL divergence across batch for each latent variable
        kl_means = torch.mean(kl_values, dim=0)
        # KL loss is sum of mean KL of each latent variable
        kl_loss = torch.sum(kl_means)

        # Record losses
        if self.global_step % self.logging_steps == 1:
            self.losses['kl_loss_cont'].append(kl_loss.item())
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)].append(kl_means[i].item())

        return kl_loss

    def _kl_multiple_discrete_loss(self, alphas):
        """
        Calculates the KL divergence between a set of categorical distributions
        and a set of uniform categorical distributions.
        Parameters
        ----------
        alphas : list
            List of the alpha parameters of a categorical (or gumbel-softmax)
            distribution. For example, if the categorical atent distribution of
            the model has dimensions [2, 5, 10] then alphas will contain 3
            torch.Tensor instances with the parameters for each of
            the distributions. Each of these will have shape (N, D).
        """
        # Calculate kl losses for each discrete latent
        kl_losses = [self._kl_discrete_loss(alpha) for alpha in alphas]

        # Total loss is sum of kl loss for each discrete latent
        kl_loss = torch.sum(torch.cat(kl_losses))

        # Record losses
        if self.global_step % self.logging_steps == 1:
            self.losses['kl_loss_disc'].append(kl_loss.item())
            for i in range(len(alphas)):
                self.losses['kl_loss_disc_' + str(i)].append(kl_losses[i].item())

        return kl_loss

    def _kl_discrete_loss(self, alpha):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)])
        log_dim = log_dim.to(self.device)
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss
"""====================== TRAIN/EVALUATE FUNCTION ======================"""

# train and evaluate function
def train(args, train_dataset, model, tokenizer, loss_function):
    """ Train the model """

    #region: ========= Setting up ========= 
    # set data sampler and dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # calculate number of total training steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # if from checkpoint
    if args.from_checkpoint:
        global_step = args.start_step
        t_total += args.start_step
    else:
        global_step = 0

    # prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},  
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]   
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)    

    # configure 16bits precision 
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)    # CHECK! IF "model.to(args.device)" or "torch.nn.DataParallel(model)" first?
    #endregion ========= Setting up =========
    
    #region: =========  Training ========= 
    # logger
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps * 1)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    # set model to train mode
    model.train()

    # set up report variables
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)


    # train iteration
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):

            # process data and setup model
            encoder_input = batch["encoder_input"].long() 
            decoder_input = batch["decoder_input"].long() 
            decoder_label = batch["decoder_label"].long()        
            encoder_attention_mask = batch["encoder_attention_mask"].long()
            decoder_attention_mask = batch["decoder_attention_mask"].long()
            encoder_input = encoder_input.to(args.device)
            decoder_input = decoder_input.to(args.device)
            decoder_label = decoder_label.to(args.device)
            encoder_attention_mask = encoder_attention_mask.to(args.device)
            decoder_attention_mask = decoder_attention_mask.to(args.device)


            ## DEBUGGING
            # model.encoder.eval()
            # model.decoder.train()
            # model.vae.train()


            # forward pass (change and edit with VAE code)
            decoder_lm_logits, latent_dist = model(encoder_input, encoder_attention_mask, decoder_input, decoder_attention_mask)

            # DEBUGGING!
            print("decoder_lm_logits.shape: " + str(decoder_lm_logits.shape))

            # calculate loss and also record loss reduction process
            loss = loss_function._loss_function(decoder_label, decoder_lm_logits, latent_dist, model.tokenizer, global_step)


            # process loss across GPUs, batches then backwards
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps


            # loss backward
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()


            # accummulte enough step, step backward
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1


                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # save model
                    loss_report = loss_function.losses
                    model.save_pretrained(args, output_dir, loss_report)                    
                    logger.info("Saving model checkpoint to %s", output_dir)
                    _rotate_checkpoints(args, checkpoint_prefix)


            # stop conditions for max_steps
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # stop conditions for max_steps    
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    #endregion ========= Training =========

    return global_step

"""====================== MAIN FUNCTION ======================"""

# main function
def main():
    
    #region ========= Set parameters ========= 
    parser = argparse.ArgumentParser()

    # dataset/save path parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--from_checkpoint", action='store_true',
                        help="To initialize model or load from a checkpoint.")    
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    
    # model parameters
    parser.add_argument("--gpt2_model_type", default=None, type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--gpt2_model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")                     
    parser.add_argument("--latent_spec_disc", default=None, type=str, required=True,
                        help="Specifications for latent discrete dimensions.")
    parser.add_argument("--latent_spec_cont", default=None, type=str, required=True,
                        help="Specifications for latent continuous dimensions.")                                                 
    parser.add_argument("--vae_hidden_dim", default=-1, type=int, required=True,
                        help="Size of latent VAE layer.")
    parser.add_argument("--vae_temperature", default=0.67, type=float, required=True,
                        help="Temperature for sampling latent dimensions.")                        
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.") 

    # training parameters
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--disc_capacity", default=None, type=str, required=True,
                        help="Capacity for latent discrete dimensions.")
    parser.add_argument("--cont_capacity", default=None, type=str, required=True,
                        help="Capacity for latent continuous dimensions.")                      
    parser.add_argument('--start_step', type=int, default=0)                        
    parser.add_argument("--frozen_layers", type=str, default='None', 
                        help="Layers to be frozen while training.")

    # other training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    
    # parsing parameters
    args = parser.parse_args()   
    #endregion ========= Set parameters =========
    
    #region ========= Checking parameters and setting up ========= 
    ## processing parameters
    # latent dimensions capacity
    cont_capacity = args.cont_capacity.split(",")
    args.cont_capacity = [float(value) for value in cont_capacity]
    disc_capacity = args.disc_capacity.split(",")
    args.disc_capacity = [float(value) for value in disc_capacity]   
    # latent dimensions specifications
    latent_spec_cont = int(args.latent_spec_cont)
    latent_spec_disc = args.latent_spec_disc.split(",")
    latent_spec_disc = [int(value) for value in latent_spec_disc]   
    args.latent_spec = {'cont': latent_spec_cont, 'disc': latent_spec_disc}

    # checking parameters
    if args.from_checkpoint is None and args.do_eval :
        raise ValueError("Cannot do evaluation without specified checkpoint.")
    if args.start_step is None and args.from_checkpoint :
        raise ValueError("Requiring determining start step from checkpoint.")    
    if args.do_train:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir and not(args.from_checkpoint):
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    

    # setting things up    
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")    # CHECK! make sure we use all 3 GPUs
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    # Set seed
    set_seed(args)
    #endregion  ========= Checking parameters and setting up =========

    #region ========= Building model ========= 
    # Building model
    gpt2_config_class, _, _ = MODEL_CLASSES[args.gpt2_model_type]
    gpt2_config = gpt2_config_class.from_pretrained(args.gpt2_model_name_or_path, cache_dir = None)
    args.gpt2_config = gpt2_config
    vae_config = VAE_config(args)
    model = VAE_GPT2(gpt2_config, vae_config, args)
    
    
    # Load from checkpoint model if run from_checkpoint
    if args.from_checkpoint == False:
        model.initialize_model(args)    # initialize model with pretrained GPT2
    else:
        model.from_pretrained(args)


    # Create loss class
    args.ignore_index = model.tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
    logger.info("args.unk_token: " + str(model.tokenizer.unk_token))
    loss_function = loss_calculation(args, vae_config)


    # Send model to GPU
    model.to(args.device)


    # Printing to check all args:
    logger.info("Arguments information: ")
    logger.info(args)    
    #endregion ========= Building model  ========= #

    #region ========= Training model ========= 
    # Training
    if args.do_train:
        
        ## DEBUGGING
        ## print model parameters 
        # logger.info("VAE")
        # for name, param in model.named_parameters():
        #     logger.info(name + ' - ' + str(param.requires_grad))


        #  freeze layers
        if args.frozen_layers is not None:
            frozen_layers = args.frozen_layers.split(" ")
            for name, param in model.named_parameters():
                if any(".{}.".format(str(frozen_layer)) in name for frozen_layer in frozen_layers):
                    logger.info("frozen params: " + name)
                    param.requires_grad = False
            
            
        # load train_dataset
        train_dataset = load_and_cache_examples(args, args.train_data_file, model.tokenizer)


        # running train function
        global_step = train(args, train_dataset, model, model.tokenizer, loss_function)
        logger.info(" global_step = %s", global_step)


        # saving model
        loss_report = loss_function.losses 
        model.save_pretrained(args, args.output_dir, loss_report)
        

        # good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))        
    #endregion ========= Training model  ========= 

if __name__ == "__main__":
    main()        

# Notes:
# - make sure loss and model all use gpu => DONE
# - chỗ do lowercase cái [pad] => DONE bỏ luôn cái lower_case vì gpt2 có thư viện chung cho cả Uncap và Cap 
# - assign UNK to end-of-text => DONE (default unk is assigned to <end-of-text>), however it's not so important since there is nothing out of vocabulary anyway, the vocab file has the whole alphabet so any word can be broken down to known tokens
# - why decoder có cái transformer ở trong list parameters => DONE because when loading, encoder dùng GPT2Model while decoder dùng GPT2LMHeadModel
# - đọc lại code từ đầu đến cuối => DONE
# - check parameters list of model => DONE
# - test when validate, the VAE is in validate mode when sampling
# - is the whole process of reshape key, value makes sense?
# - check chỗ [UNK] sau khi đã sửa lại cái +2 thành +3 => DONE
# - are the masks created correctly? 
# - check if decoder_lm_logits[0] is batch_size => DONE
# - try running with multiple GPUs => DONE run for just 1 for now
# - these values are not used: gpt2_class, tokenizer_class => DONE, because we import those gpt2_class, tokenizer_class right in model.py already

# To do:
# => check whole model, address all above mentioned problems
# => list all parameters of models

