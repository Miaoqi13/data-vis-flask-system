import os

def export_data(df):

    os.makedirs("static/export", exist_ok=True)

    file_path = "static/export/result.xlsx"

    df.to_excel(
        file_path,
        index=False
    )

    return file_path
if __name__ == "__main__":

    from modules.upload import load_data
    from modules.clean import clean_data

    df = load_data("datasets/Superstore.csv")

    clean_result = clean_data(df)

    clean_df = clean_result["df"]

    path = export_data(clean_df)

    print("导出成功：", path)