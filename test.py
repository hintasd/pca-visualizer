import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

st.title("高維數據(mnist) 可視化工具")

uploaded_file = st.file_uploader("上傳 MNIST CSV 檔案", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if df.shape[1] < 785:
        st.error("CSV 檔案格式錯誤，預期至少有 1 欄標籤 + 784 欄像素。")
    else:
        # 1. 讀取原始標籤與像素
        label_ints = df.iloc[:, 0].astype(int)
        images = df.iloc[:, 1:].astype(float).to_numpy()

        # 2. 取得所有出現過的數字，並讓使用者多選
        all_digits_int = sorted(label_ints.unique())
        selected_digits = st.multiselect(
            "選擇要顯示的數字",
            options=all_digits_int,
            default=all_digits_int[:5]  # 預設選前五個
        )

        if selected_digits:
            # 3. 篩選出使用者想看的數字，並同時保留篩選後的索引
            mask = label_ints.isin(selected_digits)
            # 篩選後的真實標籤 Series（保留原始 DataFrame 的 index）
            series_labels = label_ints[mask]
            # 篩選後的影像 array
            filtered_images = images[mask]

            # 將 series_labels reset index 以方便後續連續編號
            filtered_labels_int = series_labels.reset_index(drop=True)
            # 篩選後對應到原始 DataFrame 的索引列表（若有需要，可以同時保留原始索引）
            orig_indices = series_labels.index.to_list()

            # 4. 用 PCA 將影像降到 2 維
            pca = PCA(n_components=2)
            components = pca.fit_transform(filtered_images)

            # 5. 準備繪圖 DataFrame，例如：PC1、PC2、label（字串）、以及篩選後的索引 idx
            df_plot = pd.DataFrame({
                "PC1": components[:, 0],
                "PC2": components[:, 1],
                "label": filtered_labels_int.astype(str),
                "idx": list(range(len(filtered_labels_int)))  # 0,1,2,...
            })

            # 6. 將 label 設為有序分類型，以便 Plotly 按數字順序排顏色
            ordered_str = [str(d) for d in all_digits_int]
            cat_type = pd.CategoricalDtype(categories=ordered_str, ordered=True)
            df_plot["label"] = df_plot["label"].astype(cat_type)

            # 7. 畫出散點圖（每個點上滑鼠會顯示篩選後 idx、label、PC1、PC2）
            fig = px.scatter(
                df_plot,
                x="PC1",
                y="PC2",
                color="label",
                category_orders={"label": ordered_str},
                labels={"label": "數字"},
                title="MNIST PCA 2D 散點圖",
                width=700,
                height=500,
                hover_data={
                    "idx": True,
                    "label": True,
                    "PC1": ":.4f",
                    "PC2": ":.4f"
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            # 8. 準備「篩選後索引」與「對應標籤」的下拉選單
            point_options = [
                {
                    "label": f"索引 {i} - 數字 {filtered_labels_int.iloc[i]}",
                    "value": i
                }
                for i in range(len(filtered_labels_int))
            ]

            selected_point = st.selectbox(
                "輸入索引查看對應手寫圖",
                options=point_options,
                format_func=lambda x: x["label"],
                index=None,
                placeholder="從列表中選擇一個索引..."
            )

            # 9. 如果使用者選了某個點，就顯示該點的影像與詳細資訊
            if selected_point is not None:
                point_index = selected_point["value"]
                # 影像像素值範圍 0~255，需要先 normalize 回 0~1
                img = filtered_images[point_index].reshape(28, 28) / 255.0

                # 用兩欄佈局：左邊放圖，右邊放文字與像素表
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.subheader("手寫數字圖像")
                    st.image(
                        img,
                        width=200
                    )

                with col2:
                    st.subheader("詳細信息")
                    st.write(f"**索引 (idx)**：{point_index}")
                    st.write(f"**真實標籤 (label)**：{filtered_labels_int.iloc[point_index]}")
                    st.write(f"**主成分1 (PC1)**：{df_plot.iloc[point_index]['PC1']:.4f}")
                    st.write(f"**主成分2 (PC2)**：{df_plot.iloc[point_index]['PC2']:.4f}")

                    # 顯示影像像素值（0~255 的整數）
                    st.subheader("圖像像素值")
                    pixel_df = pd.DataFrame((img * 255).astype(int))
                    st.dataframe(pixel_df, height=300)
            else:
                st.info("請從下拉選單中選擇一個數據點以查看對應的手寫數字。")
        else:
            st.info("請至少選擇一個數字類別以顯示。")
