# Hướng dẫn sử dụng

## 1. Tiền xử lý dữ liệu

Clone repository từ GitHub: https://github.com/c1khoa/CS313-data-mining

### Cấu trúc thư mục dự án

```
project/
├── notebooks/              # Chứa các Jupyter Notebook
│   ├── crawl/             # Notebook để thu thập dữ liệu (data crawling)
│   ├── eda/               # Notebook phân tích dữ liệu khám phá
│   └── modeling/          # Notebook thử nghiệm và huấn luyện mô hình
├── models/                # Chứa mô hình đã huấn luyện và các định nghĩa mô hình
├── src/                   # Chứa source code
│   └── preprocessing/     # Script xử lý dữ liệu
├── data/                  # Chứa dữ liệu dự án
│   ├── raw/              # Dữ liệu gốc, chưa qua xử lý
│   └── preprocessed/     # Dữ liệu đã được tiền xử lý (tạo sau khi chạy script)
```

### Chạy tiền xử lý
Tải thư viện từ requirements ```pip install -r requirements.txt```

Chạy file `src/preprocessing/main_script.py` để tạo dữ liệu đã tiền xử lý trong folder `data/preprocessed`

```bash
python src/preprocessing/main_script.py
```

## 2. Chuẩn bị dữ liệu

### Định nghĩa các thông số

#### Dataset:
- `time_in`: Số ngày input cần thiết để mô hình dự đoán ngày tiếp theo (mặc định: 30)
- `time_out`: Số ngày mô hình sẽ dự đoán ra (mặc định: 14)

#### Dataloader:
- `path`: Đường dẫn đến folder `data/preprocessed` (cần setting cho phù hợp với cấu trúc thư mục của anh em)
- `time_in`: 30
- `time_out`: 14
- `batch_size_1`: 8
- `batch_size_2`: 16
- `num_workers`: 4
- `pin_memory`: True
- `drop_last`: False

### Sử dụng DataLoader

Load hàm `get_dataloaders` từ file `src/data_module.py` và sử dụng để train.

#### Cấu trúc giá trị trả về:

```python
{
"stage_1": { 
        "train": <DataLoader object>,
        "val":   <DataLoader object>,
        "test":  <DataLoader object>
        }
"stage_2": { 
        "train": <DataLoader object>,
        "val":   <DataLoader object>,
        "test":  <DataLoader object>
        }
}
```

#### Ví dụ sử dụng:

```python
from src.data_module import get_dataloaders

dataloaders = get_dataloaders(
    path="data/preprocessed",
    time_in=30,
    time_out=14,
    batch_size_1=8,
    batch_size_2=16,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)

# Sử dụng dataloader
train_loader_features = dataloaders["train"]
val_loader_features = dataloaders["val"]
```
- Input shape: torch.Size([B, time_in, 24, 35, 35])
- Target shape: torch.Size([B, time_out, 1, 35, 35])
## 3. Training

### Hướng dẫn training:
1. **Forward trực tiếp 2 model**
    - model 1:
        - input (B, time_in, C, H, W)
        - output (B, time_out, C, H, W)
    - model 2:
        - input (B, time_out, C, H, W)
        - output (B, 1, C, H, W)

2. **Sử dụng Jupyter Notebook** để thực hiện training
3. **Tổ chức file**: Có thể đặt file notebook ở bất kỳ đâu để tiện ghi đường dẫn khi làm việc
4. **Sau khi hoàn thành**:
   - Chuyển file notebook về folder `notebooks/modeling/`
   - Mỗi file notebook nên train một cặp model
   - Lưu file model đã train vào folder `models/`

### Quy trình training:

```python
# 1. Load dữ liệu
dataloaders = get_dataloaders(...)

# 2. Khởi tạo model
model = YourModel()

# 3. Training
for epoch in range(num_epochs):
    # Training loop
    for batch in dataloaders["FolderTimeSeriesDataset"]["train"]:
        # Your training code
        pass

# 4. Lưu model
torch.save(model.state_dict(), 'models/your_model_name.pth')
```

### Lưu ý:
- Forward trực tiếp 2 model rồi backward cập nhật trọng số, không train song song.
- Sau khi training xong, nhớ lưu model vào folder `models/`
- Đặt tên file model có ý nghĩa để dễ quản lý
- Ghi chú lại các hyperparameters và kết quả trong notebook

## 4. Lựa chọn model
1. Baseline: ConvLSTM + U-Net

2. ConvGRU + FPN: nhẹ, ổn định, train nhanh

3. TrajGRU + UNet++: thông minh hơn, học được chuyển động mưa/gió, phù hợp với dữ liệu khí tượng.

4. PredRNN++ + UNet++: nặng nhất nhưng khá mạnh
