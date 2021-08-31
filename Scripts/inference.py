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
import pandas as pd

from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm, trange
from transformers import (AdamW, get_linear_schedule_with_warmup, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from utils import TextDataset, load_and_cache_examples, set_seed, _rotate_checkpoints
from model import VAE_config, VAE_GPT2
from latent_traversals import LatentTraverser



logger = logging.getLogger(__name__)
MODEL_CLASSES = {'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)}

"""====================== METHODS DEFINITIONS ======================"""

def get_embeddings(model, args):

    # Construct cached file name 
    _, sentences_filename = os.path.split(args.sentences_file)    
    model_directory = args.output_dir   
    cache_embeddings_file = os.path.join(model_directory, 'cached_inference_' + sentences_filename)
    logger.info("cache_embeddings_file: " + cache_embeddings_file)


    # Check if embeddings had been extracted in cached file, read cached file if exists
    if os.path.exists(cache_embeddings_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cache_embeddings_file)
        with open(cache_embeddings_file, 'rb') as handle:
            # sentences_texts, sentences_embeddings = pickle.load(handle)        
            sentences_ids, sentences_texts, sentences_embeddings = pickle.load(handle)              
    else: # If no cached file 
        # reading file
        print("reading input file.")
        sentences = []
        with open(args.sentences_file, encoding="utf-8") as file_p:
            for sentence in file_p: 
                sentence = sentence.strip("\n")
                sentences.append(sentence) 
        logger.info("Read {} sentences.".format(str(len(sentences))))        


        # defining method for creating mask and truncating
        def create_attention_mask(sentence_length, seq_length, gpt2_config, mask_type):
            # args:
            #   sentence_length is length of real text, from [SOS] to <|endoftext|>
            #   seq_length is length with [PAD] (32, 64, 128, ...)
            
            if mask_type == "encoder_mask":
                mask_one_head = np.zeros([seq_length, seq_length])
                mask_one_head[:, :sentence_length] = 1   # the attention of [PAD] is also the input sentence
                mask_all_heads = [mask_one_head] * gpt2_config.n_head
                mask_all_heads = np.array(mask_all_heads)
            if mask_type == "decoder_mask":
                # attention, the triangular matrix is [seq_length,seq_length+1] becasuse the original has one more "past" token
                mask_one_head = np.tril(np.ones([seq_length,seq_length+1]),1)
                mask_all_heads = [mask_one_head] * gpt2_config.n_head   
                mask_all_heads = np.array(mask_all_heads)
            return mask_all_heads                        
        def truncating_padding_sentence(tokens, block_size):
                if (len(tokens) > block_size):
                    original_tokens_len = block_size
                    tokens = tokens[:block_size]
                else:
                    original_tokens_len = len(tokens)
                    tokens = tokens + ["[PAD]"]*(block_size - len(tokens))
                return tokens, original_tokens_len    


        # Processing text
        print("tokenizing and processing.")
        converted_tokenized_sentences = []    
        attention_mask_sentences = []
        for sentence in sentences:

            # tokenizing text
            sentence_text = sentence
            sentence_tokenized = model.tokenizer.tokenize(sentence_text)

            # encoder_input
            encoder_input = sentence_tokenized
            encoder_input, encoder_input_len = truncating_padding_sentence(encoder_input, args.block_size)
            encoder_input = model.tokenizer.convert_tokens_to_ids(encoder_input)
            encoder_input = np.array(encoder_input)

            # encoder_attention_mask
            encoder_attention_mask = create_attention_mask(encoder_input_len, args.block_size, args.gpt2_config, "encoder_mask")
            
            # append to lists
            converted_tokenized_sentences.append(encoder_input)
            attention_mask_sentences.append(encoder_attention_mask)
                

        # Extract embeddings
        batch_size = args.batch_size
        print("get embeddings with batch size {}.".format(str(batch_size)))
        sentences_ids = []
        sentences_embeddings = []
        sentences_texts = []
        for i in tqdm(list(range(0, len(sentences), batch_size))):

            # get batch
            sentences_text_batch = sentences[i:min(i+batch_size, len(sentences))]
            converted_tokenized_sentence = converted_tokenized_sentences[i:min(i+batch_size, len(sentences))]
            attention_mask_sentence = attention_mask_sentences[i:min(i+batch_size, len(sentences))]

            # convert to tensor, move to device
            converted_tokenized_sentence = torch.tensor(converted_tokenized_sentence).long().to(args.device) 
            attention_mask_sentence = torch.tensor(attention_mask_sentence).long().to(args.device) 

            # get embedding
            latent_dist = model.encode(converted_tokenized_sentence, attention_mask_sentence)
            sentence_embedding = model.reparameterize(latent_dist)

            # append
            if sentences_embeddings == []:
                sentences_ids = list(range(i,min(i+batch_size, len(sentences))))
                sentences_embeddings = sentence_embedding.detach().cpu().numpy()
                sentences_texts = sentences_text_batch
            else:    
                sentences_ids.extend(list(range(i,min(i+batch_size, len(sentences)))))
                sentences_embeddings = np.concatenate((sentences_embeddings, sentence_embedding.detach().cpu().numpy()), axis = 0)
                sentences_texts.extend(sentences_text_batch)
        logger.info("sentences_embeddings.shape: " + str(sentences_embeddings.shape))
    

        # Save to file
        with open(cache_embeddings_file, 'wb') as handle:
            pickle.dump([sentences_ids, sentences_texts, sentences_embeddings], handle)    
            logger.info("dump file to: {}, number of records: {}, file size: {}".format(cache_embeddings_file, str(len(sentences_embeddings)), str(os.path.getsize(cache_embeddings_file))))
            
            # Save as csv file
            cache_embeddings_file_csv, _ = os.path.splitext(cache_embeddings_file)
            cache_embeddings_file_csv += '.csv'
            logger.info("also save to csv file: {}".format(cache_embeddings_file_csv))
            text_df = pd.DataFrame(sentences_texts)
            sentences_ids = [str(sentences_id) for sentences_id in sentences_ids]
            text_df.index = sentences_ids
            text_df.insert(0, 'message_id', sentences_ids)
            text_df.columns = ["message_id", "message"]
            embeddings_df = pd.DataFrame(sentences_embeddings)
            embeddings_df.index = sentences_ids
            embeddings_df.columns = ["COMPONENT_" + str(i) for i in range(sentences_embeddings.shape[1])]
            df = text_df.merge(embeddings_df, left_on = text_df.index, right_on = embeddings_df.index)
            df = df.drop(columns=['key_0'])
            df.message_id = df.message_id.astype(str)
            df.to_csv(cache_embeddings_file_csv, index = False)
                
def inference_test_1(model, args):
    print(" === inference_test_1 ===")
    print("Task: generate embeddings from training data and analyze embeddings + reconstruction of training data")
    

    return

def inference_test_2(model, args):
    print(" === inference_test_2 ===")
    print("Task: generate text from latent dimensions")
    
    # method to decode text from logits
    def decode_to_text(decoder_lm_logits):
        # Decode to text
        predictions = torch.nn.functional.softmax(decoder_lm_logits, dim = -1)
        for prediction in predictions:
            prediction_text = tokenizer.decode(torch.argmax(prediction, dim=-1).tolist(), clean_up_tokenization_spaces=True)
            first_endoftext = prediction_text.find("<|endoftext|>") 
            logger.info("decoded_text: " + str(prediction_text[:(first_endoftext + 13) if first_endoftext>0 else len(prediction_text)])) 
            return

    # set sampling object
    latent_traverser = LatentTraverser(model.latent_spec)

    # running experiments
    if args.experiment == "sampling_experiment":

        # get sampled embeddings
        latent_traverser.sample_prior = True
        size = [args.size,args.size]
        prior_samples = latent_traverser.traverse_grid(size=size)

        # inference from embeddings
        decoder_lm_logits = model.inference(prior_samples, args)

        # decode to text
        decode_to_text(decoder_lm_logits)
        return

    if args.experiment == "cont_latent_traversals_experiment":

        # get sampled embeddings
        latent_traverser.sample_prior = args.sample_prior
        size = args.size
        latent_samples = latent_traverser.traverse_line(cont_idx=args.cont_idx,
                                                             disc_idx=None,
                                                             size=size)

        # inference from embeddings
        decoder_lm_logits = model.inference(latent_samples, args)

        # decode to text
        decode_to_text(decoder_lm_logits)
        return

    if args.experiment == "dist_latent_traversals_experiment":

        # get sampled embeddings
        latent_traverser.sample_prior = args.sample_prior
        size = args.size
        latent_samples = latent_traverser.traverse_line(cont_idx=None,
                                                             disc_idx=args.disc_idx,
                                                             size=size)

        # inference from embeddings
        decoder_lm_logits = model.inference(latent_samples, args)

        # decode to text
        decode_to_text(decoder_lm_logits)
        return
    return


"""====================== MAIN FUNCTION ======================"""

# main function
def main():

    #region ========= Set parameters ========= 
    parser = argparse.ArgumentParser()

    # dataset/save path parameters
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--primals_file", default=None, type=str,
                        help="An optional input generate data file.")
    parser.add_argument("--sentences_file", default=None, type=str,
                        help="An optional input generate data file.")
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

    # generating parameters
    parser.add_argument("--inference_test", default=0, type=int)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--generate_num", type=int, default=None)
    parser.add_argument("--generate_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--batch_size", type=int, default=None)

    # other generating parameters
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")    
    
    # parsing parameters
    args = parser.parse_args()
    #endregion ========= Set parameters =========
    
    #region ========= Checking parameters and setting up =========
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
    
    
    # Load from checkpoint model
    model.from_pretrained(args)


    # Send model to GPU
    model.to(args.device)    


    # Logging info
    logger.info("Inference parameters %s", args)
    

    # Printing to check all args:
    logger.info("Arguments information: ")
    logger.info(args)
    #endregion ========= Building model =========

    #region ========= Testing and inferencing =========
    # set model to eval() mode
    model.eval()

    # running tests
    if args.inference_test == 1:        
        inference_test_1(model, args)    
    if args.inference_test == 2:            
        inference_test_2(model, args)
    #endregion ========= Testing and inferencing =========
    
if __name__ == "__main__":
    main()        


# - make sure that the embeddings are get by model.eval() not model.train()
# - make sure that the text processing and masks are correct








