import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import simpledialog, messagebox

# 1. Đọc dữ liệu MovieLens 100k từ thư mục "ml-100k"
columns = ['user_id', 'item_id', 'rating', 'timestamp']

data_path_train = 'd:/H.U.S.T/Intro to AI/Project/APP/ml-100k/ub.base'  # Đảm bảo thư mục này chứa dữ liệu u.data
train_data = pd.read_csv(data_path_train, sep='\t', names=columns, encoding='latin-1')

data_path_test = 'd:/H.U.S.T/Intro to AI/Project/APP/ml-100k/ub.test'  # Đảm bảo thư mục này chứa dữ liệu u.data
test_data = pd.read_csv(data_path_test, sep='\t', names=columns, encoding='latin-1')

movies_set_raw = pd.read_csv("ml-100k/u.item", encoding="latin-1", sep="|", names=["movie_id", "movie_name", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10", "col11", "col12", "col13", "col14", "col15", "col16", "col17", "col18", "col19", "col20", "col21", "col22", "col23", "col24"])
movies_set = movies_set_raw.iloc[:,:2]

# Tạo từ điển ánh xạ movie_id -> movie_name
movie_dict = pd.Series(movies_set['movie_name'].values, index=movies_set['movie_id']).to_dict()

# print(df.head(5))

# 2. Chuyển đổi dữ liệu thành ma trận user-item
n_users = train_data['user_id'].nunique()
n_items = train_data['item_id'].nunique()

# Tạo ma trận user-item
user_item_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# print(user_item_matrix)

# Tạo ma trận huấn luyện và kiểm tra
train_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
test_matrix = test_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# print(train_matrix)
# print(test_matrix)

# 3. Khớp số lượng cột giữa train và test set
# Ta phải đảm bảo test_matrix có đủ các cột giống train_matrix, nếu thiếu cột thì điền bằng 0
test_matrix = test_matrix.reindex(columns=train_matrix.columns, fill_value=0)

# 4. Huấn luyện mô hình SVD (TruncatedSVD)
svd = TruncatedSVD(n_components=1000, random_state=42)
train_matrix_svd = svd.fit_transform(train_matrix)

# 5. Dự đoán giá trị cho test set
test_matrix_pred = svd.transform(test_matrix)

# 6. Tính toán RMSE, Accuracy, F1, Recall
# Ta sẽ tính RMSE giữa ma trận gốc và ma trận dự đoán của test set
y_true = test_matrix.values.flatten()
y_pred = np.dot(test_matrix_pred, svd.components_).flatten()


# Loại bỏ các giá trị 0 (chưa đánh giá) trong cả y_true và y_pred
mask = y_true != 0
y_true = y_true[mask]
y_pred = y_pred[mask]

# Tính RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Đánh giá mô hình (các metric)
accuracy_score_value = accuracy_score(y_true, y_pred.round())
f1 = f1_score(y_true, y_pred.round(), average='weighted')
recall = recall_score(y_true, y_pred.round(), average='weighted')

print(f'RMSE: {rmse}')
print(f'Accuracy: {accuracy_score_value}')
print(f'F1 Score: {f1}')
print(f'Recall: {recall}')

#6 Giao diện Tkinter để đánh giá phim và nhận đề xuất
class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation")
        self.root.geometry("600x400")

        self.movie_list = list(set(train_data['item_id']))
        self.user_ratings = {}

        self.label = tk.Label(root, text="Đánh giá 10 phim bạn thích", font=("Arial", 16))
        self.label.pack(pady=10)

        self.button = tk.Button(root, text="Đánh giá phim", command=self.get_ratings)
        self.button.pack(pady=10)

        self.recommend_button = tk.Button(root, text="Đề xuất phim", command=self.recommend_movies)
        self.recommend_button.pack(pady=10)

    def get_ratings(self):
        for i in range(10):
            movie_id = np.random.choice(self.movie_list)
            movie_name = movie_dict.get(movie_id, f"Phim {movie_id}")  # Tra cứu tên phim
            rating = simpledialog.askinteger("Đánh giá phim", f"Đánh giá phim {movie_name} (1-5):")
            if rating is not None and 1 <= rating <= 5:
                self.user_ratings[movie_id] = rating

        messagebox.showinfo("Thông báo", "Cảm ơn bạn đã đánh giá 10 phim!")

    def recommend_movies(self):
        if len(self.user_ratings) < 10:
            messagebox.showerror("Lỗi", "Vui lòng đánh giá 10 phim trước khi nhận đề xuất!")
            return
        
        # Cập nhật bộ test với các đánh giá của người dùng (sử dụng user_id giả định)
        user_id = 1  # Giả sử người dùng đánh giá là user_id = 1
        test_set = [(user_id, movie_id, rating) for movie_id, rating in self.user_ratings.items()]
        
        # Dự đoán cho các phim chưa đánh giá của người dùng
        recommended_movies = []
        for movie_id in self.movie_list:
            movie_index = movie_id - 1  # Điều chỉnh nếu item_id bắt đầu từ 1
            
            # Kiểm tra nếu movie_index hợp lệ
            if movie_index >= len(svd.components_[0]):
                continue  # Bỏ qua nếu movie_index vượt quá giới hạn

            # Tạo một vector người dùng giả định cho người dùng đầu tiên (hoặc bạn có thể tính toán cho người dùng cụ thể)
            user_vector = train_matrix.iloc[0].values  # Giả sử là người dùng đầu tiên, có thể thay thế theo yêu cầu
            
            # Tính toán điểm số dự đoán cho phim từ vector người dùng và ma trận components_
            prediction = np.dot(svd.transform([user_vector]), svd.components_[:, movie_index])
            recommended_movies.append((movie_id, prediction))

        recommended_movies.sort(key=lambda x: x[1], reverse=True)
        top_10_movies = recommended_movies[:10]

        result = "Top 10 phim đề xuất cho bạn:\n"
        for movie_id, rating in top_10_movies:
            movie_name = movie_dict.get(movie_id, f"Phim {movie_id}")  # Tra cứu tên phim
            result += f"Phim: {movie_name}, Đánh giá dự đoán: {rating.item():.2f}\n"

        messagebox.showinfo("Đề xuất phim", result)

# Khởi động giao diện Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()