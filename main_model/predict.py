"""Independent local checkpoint inference; no evaluation or text logging."""
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from common import read_rows, probabilities, write_predictions, load_tokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-length', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.batch_size < 1 or (args.precision == 'fp16' and args.device != 'cuda'):
        raise ValueError('Invalid batch size/FP16 device')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(args.device)
    tokenizer = load_tokenizer(args.model_dir)
    length = args.max_length or getattr(model.config, 'detection_max_length', 256)
    if not 1 <= length <= 512:
        raise ValueError('max_length must be 1..512')
    rows = read_rows(args.input)
    probs = probabilities(model, tokenizer, rows, length, args.batch_size, args.precision)
    write_predictions(args.output, rows, probs)
    print('PASS: ordered sample_id,p_machine; class index 1=machine; finite probabilities in [0,1].')

if __name__ == '__main__':
    main()
