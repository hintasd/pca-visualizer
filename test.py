import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

st.title("MNIST Viewer")

f = st.file_uploader("上傳 MNIST CSV 檔案", type=["csv"])
if f is not None:
    d = pd.read_csv(f)
    if d.shape[1] < 785:
        st.error("CSV 檔案格式錯誤")
    else:
        y = d.iloc[:, 0].astype(int)
        X = d.iloc[:, 1:].astype(float).to_numpy()

        dg = sorted(y.unique())
        s = st.multiselect("選擇數字", options=dg, default=dg[:5])

        if s:
            m = y.isin(s)
            y_f = y[m].reset_index(drop=True)
            X_f = X[m]
            idx = y[m].index.to_list()

            pca = PCA(2)
            c = pca.fit_transform(X_f)

            df = pd.DataFrame({
                "x": c[:, 0],
                "y": c[:, 1],
                "l": y_f.astype(str),
                "i": list(range(len(y_f)))
            })

            df["l"] = df["l"].astype(pd.CategoricalDtype(categories=[str(d) for d in dg], ordered=True))

            fig = px.scatter(
                df, x="x", y="y", color="l",
                category_orders={"l": [str(d) for d in dg]},
                labels={"l": "數字"},
                title="PCA 2D 散點圖",
                width=700, height=500,
                hover_data={"i": True, "l": True, "x": ":.4f", "y": ":.4f"}
            )
            st.plotly_chart(fig, use_container_width=True)

            opts = [{"label": f"{i} - {y_f.iloc[i]}", "value": i} for i in range(len(y_f))]
            sel = st.selectbox("選擇索引", options=opts, format_func=lambda x: x["label"], index=None, placeholder="選擇一筆資料")

            if sel is not None:
                i = sel["value"]
                img = X_f[i].reshape(28, 28) / 255.0
                c1, c2 = st.columns([1, 3])

                with c1:
                    st.image(img, width=200)

                with c2:
                    st.write(f"索引: {i}")
                    st.write(f"標籤: {y_f.iloc[i]}")
                    st.write(f"PC1: {df.iloc[i]['x']:.4f}")
                    st.write(f"PC2: {df.iloc[i]['y']:.4f}")
                    st.dataframe(pd.DataFrame((img * 255).astype(int)), height=300)
        else:
            st.info("請選擇至少一個數字")
