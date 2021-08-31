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
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, args):
        
        # reading data file
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_' + str(args.block_size) + '_' + filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)       


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
                
            
            # reading file
            self.examples = []
            with open(file_path, encoding="utf-8") as file_p:
                for sentence in file_p: 
                    
                    # tokenizing text
                    sentence_text = sentence
                    sentence_tokenized = tokenizer.tokenize(sentence_text)
                    
                    # encoder_input
                    encoder_input = sentence_tokenized
                    encoder_input, encoder_input_len = truncating_padding_sentence(encoder_input, args.block_size)
                    encoder_input = tokenizer.convert_tokens_to_ids(encoder_input)
                    encoder_input = np.array(encoder_input)
                    # decoder_input
                    decoder_input = ["[SOS]"] + sentence_tokenized
                    decoder_input, decoder_input_len = truncating_padding_sentence(decoder_input, args.block_size)
                    decoder_input = tokenizer.convert_tokens_to_ids(decoder_input)
                    decoder_input = np.array(decoder_input)
                    # decoder_output
                    decoder_label = sentence_tokenized + ["<|endoftext|>"]
                    decoder_label, _ = truncating_padding_sentence(decoder_label, args.block_size)
                    decoder_label = tokenizer.convert_tokens_to_ids(decoder_label)
                    decoder_label = np.array(decoder_label)

                    # encoder_attention_mask
                    encoder_attention_mask = create_attention_mask(encoder_input_len, args.block_size, args.gpt2_config, "encoder_mask")
                    # decoder_attention_mask
                    decoder_attention_mask = create_attention_mask(decoder_input_len, args.block_size, args.gpt2_config, "decoder_mask")

                    # DEBUGGING!
                    logger.info("sentence_tokenized: " + str(sentence_tokenized))
                    logger.info("encoder_input: " + str(encoder_input))
                    logger.info("encoder_attention_mask: " + str(encoder_attention_mask[0]))
                    logger.info("encoder_attention_mask: " + str(np.sum(encoder_attention_mask[0][0])))
                    logger.info("[SOS] + sentence_tokenized: " + str(["[SOS]"] + sentence_tokenized))
                    logger.info("decoder_input: " + str(decoder_input))
                    logger.info("decoder_attention_mask: " + str(decoder_attention_mask[0]))

                    # append to examples list
                    training_sentence = dict({"sentence_text": sentence_text, "encoder_input": encoder_input, "encoder_attention_mask": encoder_attention_mask ,"decoder_input": decoder_input, "decoder_attention_mask": decoder_attention_mask, "decoder_label": decoder_label})  
                    self.examples.append(training_sentence)
                  
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]
    
def load_and_cache_examples(args, file_path, tokenizer):
    dataset = TextDataset(tokenizer, file_path=file_path, args=args)
    return dataset    

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)   

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
