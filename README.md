# Dự đoán giá nhà bằng MLP

Bài tập nhóm xây dựng mô hình **Multilayer Perceptron (MLP)** bằng **PyTorch** để dự đoán giá nhà, thực hiện theo quy trình **CRISP-DM**.

## Dữ liệu

Sử dụng dữ liệu từ cuộc thi [House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data):

- `train.csv`: 1.460 mẫu có nhãn `SalePrice`.
- `test.csv`: 1.459 mẫu không có nhãn.
- Biến mục tiêu: giá bán nhà `SalePrice` (USD).

## Phương pháp

1. Xác định bài toán và mục tiêu.
2. Khảo sát dữ liệu.
3. Xử lý giá trị thiếu, One-Hot Encoding và chuẩn hoá.
4. Huấn luyện MLP với ba lớp ẩn: 128 → 64 → 32.
5. Đánh giá bằng RMSE trên thang log và giá gốc.
6. Lưu trọng số và xuất kết quả dự đoán.

## Cách chạy

Cài đặt thư viện:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch notebook
```

Đặt `train.csv` và `test.csv` cùng thư mục với `lab02.ipynb`, sau đó mở notebook:

```bash
jupyter notebook lab02.ipynb
```

Chạy lần lượt các cell từ trên xuống.

## Kết quả

Kết quả được lưu trong notebook:

| Chỉ số | Giá trị |
|---|---:|
| RMSE trên log(1 + SalePrice) | 0,1623 |
| RMSE trên giá gốc | 34.022,46 USD |

Lưu ý: phiên bản hiện tại chuẩn hoá dữ liệu trước khi chia train/validation, gây data leakage. Cần sửa và đánh giá lại; kết quả cũng có thể thay đổi giữa các lần chạy do chưa cố định random seed.

## Đầu ra

- `mlp_house_price.pth`: trọng số mô hình.
- `submission_pytorch.csv`: giá dự đoán cho tập test.

## Phân công

| Thành viên | Công việc |
|---|---|
| Nguyễn Hoàng Long | Đọc paper, xác định bài toán. |
| Lê Phước Thành | Khảo sát và tiền xử lý dữ liệu. |
| Hà Tuấn Duy | Xây dựng và huấn luyện MLP. |
| Đường Minh Đức | Đánh giá mô hình và tổng hợp báo cáo. |

## Tài liệu tham khảo

De Cock, D. (2011). [Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project](https://jse.amstat.org/v19n3/decock.pdf). Journal of Statistics Education, 19(3).
