# import libraries
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import logging
import torch.nn.functional as F
import os
import numpy as np
import pickle
from utils import top_k_top_p_filtering
EPS = 1e-12

logger = logging.getLogger(__name__)


# ===================== classes of model ===================== #
# define VAE_config
class VAE_config():
    def __init__(self, args):

        # Parameters 
        self.is_continuous = 'cont' in args.latent_spec
        self.is_discrete = 'disc' in args.latent_spec
        self.latent_spec = args.latent_spec
        self.temperature = args.vae_temperature
        self.hidden_dim = args.vae_hidden_dim  
        self.device = args.device


# define class VAE
class VAE(nn.Module):
    def __init__(self, gpt2_config, vae_config):
        super(VAE, self).__init__()

        # Parameters 
        self.is_continuous = vae_config.is_continuous
        self.is_discrete = vae_config.is_discrete
        self.latent_spec = vae_config.latent_spec
        self.temperature = vae_config.temperature
        self.hidden_dim = vae_config.hidden_dim
        self.device = vae_config.device


        # Calculate dimensions of latent distribution
        self.latent_cont_dim = 0
        self.latent_disc_dim = 0
        self.num_disc_latents = 0
        if self.is_continuous:
            self.latent_cont_dim = self.latent_spec['cont']
        if self.is_discrete:
            self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
            self.num_disc_latents = len(self.latent_spec['disc'])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim        


        ## Encoder defining
        # Layers to transform features to hidden
        self.features_to_hidden = nn.Sequential(
            nn.Linear(gpt2_config.n_embd * 2, self.hidden_dim),
            nn.ReLU()
        )
         # Encode parameters of latent distribution
        if self.is_continuous:
            self.fc_mean = nn.Linear(self.hidden_dim, self.latent_cont_dim)
            self.fc_log_var = nn.Linear(self.hidden_dim, self.latent_cont_dim)
        if self.is_discrete:
            # Linear layer for each of the categorical distributions
            fc_alphas = []
            for disc_dim in self.latent_spec['disc']:
                fc_alphas.append(nn.Linear(self.hidden_dim, disc_dim))
            self.fc_alphas = nn.ModuleList(fc_alphas)       


        ## Decoder defining
        # Layers to transform latent to hidden
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU()
        )    
        # Layers to hidden latent to features
        self.hidden_to_features = nn.Sequential(
            nn.Linear(self.hidden_dim, gpt2_config.n_embd * 2),
            nn.ReLU()
        )    

    def encode(self, features):
            """
            Encodes features to latent code
            """
            batch_size = features.size()[0]

            # Encode image to hidden features
            hidden = self.features_to_hidden(features.view(batch_size, -1))

            # Output parameters of latent distribution from hidden representation
            latent_dist = {}

            if self.is_continuous:
                latent_dist['cont'] = [self.fc_mean(hidden), self.fc_log_var(hidden)]

            if self.is_discrete:
                latent_dist['disc'] = []
                for fc_alpha in self.fc_alphas:
                    latent_dist['disc'].append(F.softmax(fc_alpha(hidden), dim=1))

            return latent_dist    

    def reparameterize(self, latent_dist):
            """
            Samples from latent distribution using the reparameterization trick.
            """
            latent_sample = []

            if self.is_continuous:
                mean, logvar = latent_dist['cont']
                cont_sample = self.sample_normal(mean, logvar)
                latent_sample.append(cont_sample)

            if self.is_discrete:
                for alpha in latent_dist['disc']:
                    disc_sample = self.sample_gumbel_softmax(alpha)
                    latent_sample.append(disc_sample)

            # Concatenate continuous and discrete samples into one large sample
            return torch.cat(latent_sample, dim=1)

    def sample_normal(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size()).normal_()
            eps = eps.to(self.device)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def sample_gumbel_softmax(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        if self.training:   # training attribute inherites from nn.Module, set to True when model.train()
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size())
            unif = unif.to(self.device)    
            EPS = 1e-12    
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            one_hot_samples = one_hot_samples.to(self.device)    
            return one_hot_samples

    def decode(self, latent_sample):
        """
        Decodes sample from latent distribution into an image.
        Parameters
        ----------
        latent_sample : torch.Tensor
            Sample from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        hidden = self.latent_to_hidden(latent_sample)
        features = self.hidden_to_features(hidden)
        return features

    def forward(self, features):
        """
        Forward pass of model.
        """
        latent_dist = self.encode(features)
        latent_sample = self.reparameterize(latent_dist)
        features_output = self.decode(latent_sample)
        
        return features_output, latent_dist


# define class VAE_GPT2
class VAE_GPT2(nn.Module):

    def __init__(self, gpt2_config, vae_config, args):
        super(VAE_GPT2, self).__init__()

        # set up encoder and decoder have the same config
        self.gpt2_config = gpt2_config
        self.vae_config = vae_config
        self.tokenizer = None
        self.encoder = None
        self.decoder = None
        self.vae = VAE(self.gpt2_config, self.vae_config)
        self.device = args.device

        # set up gpt2_config
        self.gpt2_config.output_hidden_states = True
        self.gpt2_config.output_past = True
        self.gpt2_config.output_attentions = True

    def initialize_model(self, args):

        # load pretrained model and tokenizer for GPT2 encoder and decoder
        encoder_path = args.gpt2_model_name_or_path
        decoder_path = args.gpt2_model_name_or_path   
        tokenizer_path = args.gpt2_model_name_or_path
        self.encoder = GPT2Model.from_pretrained(encoder_path, from_tf=bool('.ckpt' in encoder_path), config=self.gpt2_config)
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_path, from_tf=bool('.ckpt' in decoder_path), config=self.gpt2_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)


        # add [UNK], [SOS] and [PAD] to tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens":["[PAD]", "[SOS]"]})
        self.tokenizer.add_special_tokens({'unk_token': '[UNK]'})   # as default unknown token will be set to <endoftext>, however we might not need to set [UNK] since GPT2 has the whole alphabet in the vocabulary so there is no unknown tokens
        self.encoder.resize_token_embeddings(len(self.tokenizer))   
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        logger.info("tokenizer size: " + str(self.tokenizer.__len__()))
        logger.info("tokenizer.decode [50256, 50257, 50258, 50259]: " + str(self.tokenizer.decode([50256, 50257, 50258, 50259])) )        

        return

    def save_pretrained(self, args, output_dir, loss_reports):

        # set up output_dir to save sub-models
        output_dir_encoder = output_dir + "/encoder/"
        output_dir_decoder = output_dir + "/decoder/"
        output_dir_tokenizer = output_dir + "/tokenizer/"
        output_dir_vae = output_dir + "/vae/"
        if not os.path.exists(output_dir_encoder):
            os.makedirs(output_dir_encoder)
        if not os.path.exists(output_dir_decoder):
            os.makedirs(output_dir_decoder)            
        if not os.path.exists(output_dir_tokenizer):
            os.makedirs(output_dir_tokenizer)
        if not os.path.exists(output_dir_vae):
            os.makedirs(output_dir_vae)
        output_dir_vae = output_dir_vae + "/vae.weights"    


        # save model
        self.encoder.save_pretrained(output_dir_encoder)
        self.decoder.save_pretrained(output_dir_decoder)
        self.tokenizer.save_pretrained(output_dir_tokenizer)
        torch.save(self.vae.state_dict(),output_dir_vae)       

        # save training args and loss record
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)
        loss_reports_file = open(output_dir + "/loss_reports.pkl", "wb")
        pickle.dump(loss_reports, loss_reports_file)
        
        return

    def from_pretrained(self, args):
        
        # loading from pre-trained
        encoder_path = args.output_dir + "/encoder/"
        decoder_path = args.output_dir + "/decoder/"
        vae_path = args.output_dir + "/vae/vae.weights"
        tokenizer_path = args.output_dir + "/tokenizer/"
        logger.info("gpt2_config: " + str(self.gpt2_config))
        self.gpt2_config.vocab_size = self.gpt2_config.vocab_size + 3 # edit when adding new tokens such as [PAD], [SOS], [UNK] 
        self.encoder = GPT2Model.from_pretrained(encoder_path, from_tf=bool('.ckpt' in encoder_path), config=self.gpt2_config)
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_path, from_tf=bool('.ckpt' in decoder_path), config=self.gpt2_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)
        self.vae.load_state_dict(torch.load(vae_path))

        # set up for evaluating
        self.encoder.eval()
        self.decoder.eval()
        self.vae.eval()

        # load training args
        training_args = torch.load(os.path.join(args.output_dir, 'training_args.bin'))
        logger.info("training_args: " + str(training_args))

        return  

    def encode(self, encoder_input_ids, encoder_attention_mask):

        # GPT-2 encoder
        _, encoder_presents, _, _ = self.encoder(input_ids = encoder_input_ids, attention_mask = encoder_attention_mask) # output(encoder_last_hidden_state, encoder_presents, encoder_hidden_states, encoder_attentions)
        # processing encoder to feed to vae
        encoder_presents = torch.stack(encoder_presents)    # encoder_presents is a tuple of layers. torch.stack transform to tensor
        batch_size = encoder_presents.shape[1]
        # process encoder_presents into 1 vector for each sample in batch
        secondToLast_encoder_presents = encoder_presents[-2]
        ## mean of all embeddings including [PAD]
        secondToLast_encoder_presents = secondToLast_encoder_presents.mean(dim = -2, keepdim = True)    # take into account [PAD], because [PAD] is also only attentioned on sentence
        # reshape to bring batch_size to be the 1st dimension
        secondToLast_encoder_presents = secondToLast_encoder_presents.reshape([batch_size, -1])  # reshape into [batch_size, hidden_size * 2] 

        # JointVAE encoder
        features = secondToLast_encoder_presents
        latent_dist = self.vae.encode(features)

        return latent_dist

    def decode(self, decoder_input_ids, decoder_attention_mask, latent_sample): 

        # JointVAR decoder
        features_output = self.vae.decode(latent_sample)

        # decoder
        secondToLast_decoder_hidden = features_output
        batch_size = secondToLast_decoder_hidden.shape[0] # CHECK!!!
        decoder_hidden = torch.zeros([self.gpt2_config.n_layer, batch_size, 2, self.gpt2_config.n_head, 1, int(self.gpt2_config.n_embd/self.gpt2_config.n_head)], device = self.device)
        secondToLast_decoder_hidden = secondToLast_decoder_hidden.reshape([batch_size, 2, self.gpt2_config.n_head, 1, int(self.gpt2_config.n_embd/self.gpt2_config.n_head)])
        decoder_hidden[-2] = secondToLast_decoder_hidden
        past = decoder_hidden

        # decoder forward pass
        decoder_lm_logits, _, _, _ = self.decoder(input_ids = decoder_input_ids, past = past, attention_mask = decoder_attention_mask) # output(decoder_lm_logits, decoder_presents, decoder_hidden_states, decoder_attentions)

        return decoder_lm_logits      

    def inference(self, latent_sample, args):
        # set batch_size to 1, we always generate for 1 latent_sample at a time
        batch_size = 1
        
        # construct hidden vector
        secondToLast_decoder_hidden = self.vae.decode(latent_sample) 
        
        # decoder 
        decoder_hidden = torch.zeros([self.gpt2_config.n_layer, batch_size, 2, self.gpt2_config.n_head, 1, int(self.gpt2_config.n_embd/self.gpt2_config.n_head)], device = self.device)
        secondToLast_decoder_hidden = secondToLast_decoder_hidden.reshape([batch_size, 2, self.gpt2_config.n_head, 1, int(self.gpt2_config.n_embd/self.gpt2_config.n_head)])
        decoder_hidden[-2] = secondToLast_decoder_hidden
        decoder_input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids("[SOS]")]*batch_size, device = self.device).reshape(batch_size,1)
        past = decoder_hidden

        # generate tokens
        generated = decoder_input_ids
        for _ in range(args.generate_length):

            # decoder forward pass
            decoder_lm_logits, _, _, _ = self.decoder(input_ids = generated, past = past, attention_mask = None)
            
            # sample from vocabulary
            decoder_lm_logits = decoder_lm_logits[:,-1,:] # get the output of the last token
            filtered_decoder_lm_logits = top_k_top_p_filtering(decoder_lm_logits, top_k=args.top_k, top_p=args.top_p)
            if args.temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_decoder_lm_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_decoder_lm_logits, dim=-1), num_samples=1)                
            generated = torch.cat((generated, next_token), dim=1)

        return generated

    def reparameterize(self, latent_dist):
        # reparameterize method by jointvae
        latent_sample = self.vae.reparameterize(latent_dist)
        return latent_sample

    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask):

        latent_dist = self.encode(encoder_input_ids, encoder_attention_mask)
        latent_sample = self.reparameterize(latent_dist)
        decoder_lm_logits = self.decode(decoder_input_ids, decoder_attention_mask, latent_sample)

        return decoder_lm_logits, latent_dist





