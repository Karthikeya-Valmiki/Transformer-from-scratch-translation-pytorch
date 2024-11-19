import torch
import torch.nn as nn
from torch. utils. data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizer import tokenizer
from tokenizers.models import WordLevel # type: ignore
from tokenizers.trainers import WordLevelTrainer # type: ignore
from tokenizers.pre_tokenizers import Whitespace # type: ignore

from pathlib import Path
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config
from config import get_weights_file_path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings


# def greedy_decode(model, source, source_mask, tokenizer_sre, tokenizer_tgt, max_len, device):
    #sos_idx = tokenizer_tgt. token_to_id('[SOS]')
    #eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # Precompute the encoder output and reuse it for every token we get from the decoder
    #encoder_output = model.encode (source, source_mask)
    #Initialize the decoder input with the sos token
    #decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    #while True:
        #if decoder_input.size(1) == max_len:
            #break
        
        # Build mask for the target (decoder input)
        #decoder_mask = causal_mask (decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Calculate the output of the decoder 
        #decoder_output = model.decode(decoder_input, decoder_mask, encoder_output)
        
    

def run_validation (model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0
    
    source_texts = []
    target_texts = []
    predicted = []
    
    # Size of the control window (just use a default value)
    console_width = 80
    
    with torch. no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch [ 'encoder mask' ].to (device)
            
            assert encoder_input. size(0) == 1, "Batch size must be 1 for validation"
            
            
## getting all sentences 
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


## Code for building tokenizer 
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        # above UNK - unknown, PAD - padded word, SOS - start of senctence, EOS - end of sentence
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer-trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = tokenizer.from_file(str(tokenizer_path))
    return tokenizer
    
  
## Code for loading the dataset
def get_ds(config):
    ds_raw = load_dataset('Helsinki-NLP/opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split= 'train')
    
    ## building tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])  
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # manual splitting of  data (90% - train, 10% - validation)
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_raw])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item ['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item ['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
        print(f'Max Length of source sentence: {max_len_src}')
        print(f'Max Length of target sentence: {max_len_tgt}')
        
        train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_ds,batch_size=1, shuffle=True)
        
        return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len,vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

## Training loop (training the model)

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (f'using device {device}' )
    
    Path(config['model_folder'].mkdir(parents=True,exist_ok= True))
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    intial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print (f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        intial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
       
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)  
        
        for epoch in range(intial_epoch, config['num_epochs']):
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch: 02d}')
            for batch in batch_iterator:
                
                encoder_input = batch['encoder_input']. to(device) # (batch, seq_len)
                decoder_input = batch['decoder_input']. to(device) # (batch, seq_len)
                encoder_mask = batch['encoder_mask']. to(device) # (Batch, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask']. to(device) # (Batch, 1, seq_len, seq_len)
                
                #Run the tensors through the transformer
                encoder_output = model.encode(encoder_input,encoder_mask) # (B, Seq_Len, d_ model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, Seq_Len, d_ model)
                proj_output = model.project(decoder_output) # (B, Seq_Len, tgt_vocab_size)
                
                label = batch ['label']. to (device) # (B, Seq_Len)
                
                # (B, Seq_Len, tgt_vocab_size) --> (B * Seq_Len, tgt_vocab_size)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
                
                # log the loss onto tensorboard
                writer.add_scalar ('train loss', loss.item(), global_step)
                writer. flush()
                
                # Backpropagate the loss
                loss. backward()
                # Update the weights
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1 
                
        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, ' {epoch: 02d} ')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer. state_dict(),
            'global_step': global_step   
        } , model_filename)
        
if __name__ == ' main_':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
    
    
    
    
    
    
    


            
            
            
    

            
                
                
                
                
                
                
            
        
    
    
    
    
    
    
        
        
  
    
    