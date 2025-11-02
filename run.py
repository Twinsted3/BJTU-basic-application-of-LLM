import torch
import csv
import random

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from config import ModelConfig
from dataset import MyDataset
from model import MyTokenizer, MyTransformer, generate_square_subsequent_mask

CONFIG = ModelConfig()  # 注册全局变量


def save_checkpoint(state, path):
    # 保存检查点
    torch.save(state, path)
    

def load_shakespeare(file_path="./src/input.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    except FileNotFoundError:
        print("错误: 'input.txt' 文件未找到。")
        print("请确保 'input.txt' 文件和您的Python脚本在同一个目录下。")
    return full_text


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # 加载数据
    text = load_shakespeare(CONFIG.data_path)
    tokenizer = MyTokenizer(text)
    data_ids = tokenizer.encode(text)
    vocab_size = tokenizer.vocab_size
    print(f"Text length: {len(text)} chars, vocab size: {vocab_size}")

    # 划分训练集测试集
    split = int(0.9 * len(data_ids))
    train_ids = data_ids[:split]
    val_ids = data_ids[split:]

    train_ds = MyDataset(train_ids, CONFIG.block_size)
    val_ds = MyDataset(val_ids, CONFIG.block_size)

    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG.batch_size, shuffle=False, drop_last=False)

    # 加载模型和优化器
    model = MyTransformer(
        vocab_size=vocab_size,
        d_model=CONFIG.d_model,
        n_layer=CONFIG.n_layer,
        n_head=CONFIG.n_head,
        d_ff=CONFIG.n_ff,
        max_len=CONFIG.block_size,
        dropout=CONFIG.dropout,
        use_positional_encoding=not CONFIG.no_positional_encoding,
        use_residual=not CONFIG.no_residual,
        use_layernorm=not CONFIG.no_layernorm
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.epochs)
    
    # 记录训练log
    print("\n--> Starting training...")
    train_log = []
    global_step = 0
    best_val_loss = float('inf')
    for epoch in range(1, CONFIG.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG.epochs}", leave=False, ascii=True)
        epoch_loss = 0.0
        iter_count = 0
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)  # (B, T, V)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            iter_count += 1
            global_step += 1
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

        train_loss = epoch_loss / max(1, iter_count)
        val_loss = evaluate(model, val_loader, device)
        train_log.append((epoch, train_loss, val_loss))
        print(f"Epoch {epoch} Train loss: {train_loss:.4f} Val loss: {val_loss:.4f}")

        # 保存
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'tokenizer': tokenizer.stoi,
        }
        ckpt_path = './ckpt.pt'
        save_checkpoint(ckpt, ckpt_path)
        torch.save(model, './model.pth')

        # csv log
        csv_path = './train_loss.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch','train_loss','val_loss'])
            writer.writerows(train_log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt, './best_ckpt.pt')

        # optionally print a short sample
        sample_text = sample(model, tokenizer, device, start_str=random.choice(list(tokenizer.stoi.keys())), length=200)
        # sample_text = sample(model, tokenizer, device, start_str='To', length=200)
        print("**** sample ****")
        print(sample_text)
        print("****************")


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='sum')
        total_loss += loss.item()
        total_count += xb.size(0) * xb.size(1)
    return total_loss / max(1, total_count)


@torch.no_grad()
def sample(model, tokenizer, device, start_str='T', length=200, temperature=1.0):
    if length > CONFIG.block_size:
        print("Warning: length is greater than block_size, truncating to block_size.")
        length = CONFIG.block_size

    model.eval()
    ids = [tokenizer.stoi.get(ch, 0) for ch in start_str]
    idx = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(length - len(start_str)):
            if idx.size(1) > model.pos_enc.pe.size(1) if getattr(model, 'pos_enc', None) is not None else False:
                # truncate to last T positions
                idx = idx[:, -model.pos_enc.pe.size(1):]
            logits = model(idx)  # (1, T, V)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
            idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0].cpu().tolist())


if __name__ == "__main__":
    # torch.random.manual_seed(77)
    if CONFIG.is_train == True:
        train()
    else:
        text = load_shakespeare(CONFIG.data_path)
        tokenizer = MyTokenizer(text)
        vocab_size = tokenizer.vocab_size

        model = MyTransformer(
            vocab_size=vocab_size,
            d_model=CONFIG.d_model,
            n_layer=CONFIG.n_layer,
            n_head=CONFIG.n_head,
            d_ff=CONFIG.n_ff,
            max_len=CONFIG.block_size,
            dropout=CONFIG.dropout,
            use_positional_encoding=True,
            use_residual=True,
            use_layernorm=True
        ).to(CONFIG.device)
        model.load_state_dict(torch.load('best_ckpt.pt', weights_only=True)["model_state_dict"])
        print("**** sample ****")
        sample_text = sample(model, tokenizer, device=CONFIG.device, start_str=random.choice(list(tokenizer.stoi.keys())), length=200)
        # sample_text = sample(model,tokenizer, device=CONFIG.device, start_str='To', length=200)  # length不能大于block_size
        print(sample_text)
        print("****************")