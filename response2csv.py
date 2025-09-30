import transformers, peft, accelerate, datasets, bitsandbytes, trl
import kagglehub
import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm.auto import tqdm
import time

def main():
    #------------Load models files from kaggle
    import os

    root_path = kagglehub.dataset_download('masao1112/models')
    print(f'Data source import complete at : {root_path}')
    # Join model directories with root_path
    nlft_model_path = os.path.join(root_path, "Models/phogpt-4b-medical-chatbot-nlft-3750-samples")
    sft_model_path = os.path.join(root_path, "Models/phogpt-4b-medical-chatbot-sft-3750-samples")
    # Check if valid path
    for name, path in [("NLFT", nlft_model_path), ("SFT", sft_model_path)]:
        if os.path.isdir(path):
            print(f"Valid {name} path: {path}")
        else:
            raise FileNotFoundError(f"Invalid {name} path: {path}")
    #-------------------------------------
    #
    #-----------// Download the dataset and preprocess it
    print("\n--- BƯỚC 1: Tải dữ liệu y tế từ HF ---")
    PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"
    dataset_name = "hungnm/vietnamese-medical-qa"
    full_dataset = load_dataset(dataset_name, split="train")
    # split the dataset into train and val sets
    train_test_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    raw_datasets = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test']
    })
    
    test_set = raw_datasets['validation']
    print(f"Đã tạo {len(test_set)} mẫu thử.")
    
    #-------------// Lấy responses từ model pretrain lẫn finetune
    print("\n--- BƯỚC 2: Tải Model gốc để tạo dữ liệu ---")
    #-------------// Base Model(Pretrain Model)
    model_name = "vinai/PhoGPT-4B-Chat"
    
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    base_model, base_tokenizer = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    ), AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.padding_side = "left"
    
    #-----------// Finetune-SFT Model 
    print("\n--- BƯỚC 2.2: Áp Finetune-SFT model vào model gốc để tạo dữ liệu ---")
    
    sft_tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    if sft_tokenizer.pad_token is None:
        sft_tokenizer.pad_token = sft_tokenizer.eos_token

    print(f"Đang áp LoRA adapter từ: {sft_model_path}")
    sft_model = PeftModel.from_pretrained(base_model, sft_model_path)
    sft_model.eval()
    
    #----------// Finetune-NLFT model
    print("\n--- BƯỚC 2.3: Áp Finetune-NLFT model vào model gốc để tạo dữ liệu ---")
    
    nlft_tokenizer = AutoTokenizer.from_pretrained(nlft_model_path)
    if nlft_tokenizer.pad_token is None:
        nlft_tokenizer.pad_token = nlft_tokenizer.eos_token

    print(f"Đang áp LoRA adapter từ: {nlft_model_path}")
    nlft_model = PeftModel.from_pretrained(base_model, nlft_model_path)
    nlft_model.eval()
    
    #----------// Q/A pipeline for SFT and NLFT
    print("Đang tạo pipeline text-generation...")
    base_qa_pipeline = get_pipeline(base_model, base_tokenizer)
    sft_qa_pipeline = get_pipeline(sft_model, sft_tokenizer)
    nlft_qa_pipeline = get_pipeline(nlft_model, nlft_tokenizer)
    
    #----------// Get Model response
    print(f"\n--- BƯỚC 3: Tạo Dữ liệu Testing ---")
    # Define output path
    base_model_output_filename = "base_model_responses_test.csv"
    sft_model_output_filename = "sft_model_responses_test.csv"
    nlft_model_output_filename = "nlft_model_responses_test.csv"
    
    # get responses  
    get_responses(test_set, base_qa_pipeline, base_model_output_filename)
    get_responses(test_set, sft_qa_pipeline, sft_model_output_filename)
    get_responses(test_set, nlft_qa_pipeline, nlft_model_output_filename)
    

def get_pipeline(model, tokenizer):
    eos_token_id = tokenizer.eos_token_id
    qa_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )
    
    return qa_pipeline

def get_responses(test_data, qa_pipeline, output_path):
    stop_sequence = "### Câu hỏi:"
    
    model_responses = []

    try:
        for i, sample in enumerate(
            tqdm(
                test_data,
                desc="Creating Testing Data",
            )
        ):

            input_text = f"### Câu hỏi: {sample['question']}\n### Trả lời:"

            # Lấy kết quả từ pipeline
            full_output = qa_pipeline(input_text)[0]["generated_text"]
        
            # Tách để lấy phần trả lời của model
            model_generated_part = full_output.split("### Trả lời:")[1]
        
            # Loại bỏ phần prompt có thể bị lặp lại ở cuối
            if stop_sequence in model_generated_part:
                model_answer = model_generated_part.split(stop_sequence)[0].strip()
            else:
                model_answer = model_generated_part.strip()

            model_responses.append(model_answer)
            
            # visualize first 3 samples
            if i < 3:
                print(f"\n--- Câu hỏi {i+1} ---")
                print(f"Câu hỏi: {sample['question']}")
                print(f"Model trả lời: {model_answer}")

    finally:
        if len(model_responses) > 0:
            print(f"\nHoàn tất xử lý hoặc bị ngắt. Đang tiến hành lưu {len(model_responses)} kết quả...")
            
            df_to_save = pd.DataFrame({'model_response': model_responses})
            df_to_save.to_csv(output_path, index=False, encoding='utf-8-sig')

            print(f"{len(model_responses)} dòng đã được lưu vào file tại đường dẫn:")
            print(f"'{output_path}'")
        else:
            print("\nKhông có kết quả nào được tạo ra để lưu.")
            

if __name__=='__main__':
    main()
