import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from TkinterDnD2 import * 
from tkinter import *

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ModelTrainingApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.title("模型训练系统")
        self.geometry("800x600")

        self.setup_ui()

        self.data = None
        self.model = None
        self.le = None  # 用于分类任务的标签编码器

    def setup_ui(self):
        # 左侧拖拽区域
        self.drop_frame = tk.LabelFrame(self, text="将CSV/XLSX文件拖拽到此处", width=300, height=400)
        self.drop_frame.pack(side="left", padx=10, pady=10)

        # 监听拖拽事件
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.load_file)

        # 模型选择
        self.model_frame = tk.LabelFrame(self, text="选择模型类型", width=300, height=150)
        self.model_frame.pack(side="top", padx=10, pady=10, fill="x")

        self.model_type_var = tk.StringVar(value="回归")
        self.regression_button = tk.Radiobutton(self.model_frame, text="回归", variable=self.model_type_var, value="回归", command=self.update_model_options)
        self.classification_button = tk.Radiobutton(self.model_frame, text="分类", variable=self.model_type_var, value="分类", command=self.update_model_options)
        self.regression_button.pack(side="left", padx=5, pady=5)
        self.classification_button.pack(side="left", padx=5, pady=5)

        self.model_selection = ttk.Combobox(self.model_frame)
        self.model_selection.pack(side="left", padx=5, pady=5, fill="x")

        # 特征选择
        self.feature_frame = tk.LabelFrame(self, text="选择特征", width=300, height=200)
        self.feature_frame.pack(side="left", padx=10, pady=10, fill="both")

        self.feature_listbox = tk.Listbox(self.feature_frame, selectmode="multiple")
        self.feature_listbox.pack(side="left", padx=5, pady=5, fill="y")

        self.target_label = tk.Label(self.feature_frame, text="选择目标特征:")
        self.target_label.pack(side="top", padx=5, pady=5)

        self.target_selection = ttk.Combobox(self.feature_frame)
        self.target_selection.pack(side="top", padx=5, pady=5, fill="x")

        # 训练按钮
        self.train_button = tk.Button(self, text="训练模型", command=self.train_model)
        self.train_button.pack(side="top", padx=10, pady=10)

        # 右侧拖拽区域
        self.prediction_frame = tk.LabelFrame(self, text="将新CSV/XLSX文件拖拽到此处进行预测", width=300, height=400)
        self.prediction_frame.pack(side="right", padx=10, pady=10)

        self.prediction_frame.drop_target_register(DND_FILES)
        self.prediction_frame.dnd_bind('<<Drop>>', self.make_prediction)

    def load_file(self, event):
        # 载入文件并显示特征名
        file_path = event.data
        self.data = pd.read_csv(file_path.strip("{}")) if file_path.endswith(".csv") else pd.read_excel(file_path.strip("{}"))

        # 清除之前的选项
        self.feature_listbox.delete(0, tk.END)
        self.target_selection.set('')

        numeric_features = self.data.select_dtypes(include=["float64", "int64"]).columns.tolist()
        string_features = self.data.select_dtypes(include=["object"]).columns.tolist()

        for feature in numeric_features:
            self.feature_listbox.insert(tk.END, feature)
        for feature in string_features:
            self.feature_listbox.insert(tk.END, feature + " (字符串)")

        self.target_selection['values'] = numeric_features + string_features

    def update_model_options(self):
        # 更新模型选项
        model_type = self.model_type_var.get()
        if model_type == "回归":
            models = ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet", "Decision Tree Regression", "Random Forest Regression"]
        else:
            models = ["Logistic Regression", "SVM", "Decision Tree Classification", "Random Forest Classification", "Gradient Boosting", "KNN Classification"]

        self.model_selection['values'] = models

    def train_model(self):
        # 获取用户选择的特征和目标
        selected_features = [self.feature_listbox.get(i) for i in self.feature_listbox.curselection()]
        target_feature = self.target_selection.get()

        if not selected_features or not target_feature:
            messagebox.showerror("错误", "请同时选择特征和目标。")
            return

        X = self.data[selected_features]
        y = self.data[target_feature]

        # 处理类别特征（如有）
        X = pd.get_dummies(X, drop_first=True)
        if self.model_type_var.get() == "分类":
            if y.dtype == 'O':
                self.le = LabelEncoder()
                y = self.le.fit_transform(y)

        # 选择模型
        model_name = self.model_selection.get()
        if model_name == "Linear Regression":
            self.model = LinearRegression()
        elif model_name == "Ridge Regression":
            self.model = Ridge()
        elif model_name == "Lasso Regression":
            self.model = Lasso()
        elif model_name == "ElasticNet":
            self.model = ElasticNet()
        elif model_name == "Decision Tree Regression":
            self.model = DecisionTreeRegressor()
        elif model_name == "Random Forest Regression":
            self.model = RandomForestRegressor()
        elif model_name == "Logistic Regression":
            self.model = LogisticRegression()
        elif model_name == "SVM":
            self.model = SVC()
        elif model_name == "Decision Tree Classification":
            self.model = DecisionTreeClassifier()
        elif model_name == "Random Forest Classification":
            self.model = RandomForestClassifier()
        elif model_name == "Gradient Boosting":
            self.model = GradientBoostingClassifier()
        elif model_name == "KNN Classification":
            self.model = KNeighborsClassifier()
        else:
            messagebox.showerror("错误", "请选择一个模型。")
            return

        # 训练模型
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # 计算评估指标
        y_pred = self.model.predict(X_test)
        if self.model_type_var.get() == "回归":
            mse = mean_squared_error(y_test, y_pred)
            self.show_training_result(mse, model_name)
            self.plot_results(y_test, y_pred, target_feature)
        else:
            accuracy = accuracy_score(y_test, y_pred)
            self.show_training_result(accuracy, model_name)
        
    def show_training_result(self, metric, model_name):
        # 显示训练结果
        result_window = tk.Toplevel(self)
        result_window.title("训练结果")
        tk.Label(result_window, text=f"模型：{model_name}", font=("Arial", 12)).pack(padx=10, pady=10)
        if self.model_type_var.get() == "回归":
            tk.Label(result_window, text=f"MSE: {metric:.4f}", font=("Arial", 12)).pack(padx=10, pady=10)
        else:
            tk.Label(result_window, text=f"准确率: {metric:.4f}", font=("Arial", 12)).pack(padx=10, pady=10)
        
        # 打印模型内部信息
        model_info = f"模型信息:\n{self.model.get_params()}" if hasattr(self.model, 'get_params') else "无法获取模型信息"
        tk.Label(result_window, text=model_info, font=("Arial", 10)).pack(padx=10, pady=10)

    def plot_results(self, y_test, y_pred, target_feature):
        # 绘制误差直方图
        result_window = tk.Toplevel(self)
        result_window.title("误差分布图")
        plt.figure(figsize=(10, 6))
        sns.histplot(abs(y_test - y_pred), bins=30, kde=True)
        plt.xlabel(f"绝对误差 ({target_feature})")
        plt.ylabel("频率")
        plt.title("误差分布")
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(plt.gcf(), master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def make_prediction(self, event):
        # 预测新数据并保存结果
        file_path = event.data.strip("{}")
        new_data = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

        new_data = pd.get_dummies(new_data, drop_first=True)
        predictions = self.model.predict(new_data)

        if self.model_type_var.get() == "分类":
            predictions = self.le.inverse_transform(predictions)  # 恢复为原来的字符串标签

        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel文件", "*.xlsx"), ("所有文件", "*.*")])
        if save_path:
            new_data['预测结果'] = predictions
            new_data.to_excel(save_path, index=False)
            messagebox.showinfo("保存成功", f"预测结果已保存到 {save_path}")

if __name__ == "__main__":
    app = ModelTrainingApp()
    app.mainloop()
