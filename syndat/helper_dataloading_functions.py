def convert_to_syndat_scores(df, only_pos=False):
    df = df[df.REPI == 1]
    for col in df.columns:
        if only_pos and col.startswith("REC_"):
            df[col] = df[col].clip(lower=0)
        if col.startswith("MASK"):
            var_name = "_".join(col.split("_")[1:])
            mask = df[col] == 0
            # Setting values to NAN where  mask is 0
            df[f'OBS_{var_name}'] = df[f'OBS_{var_name}'].mask(mask)
            df[f'REC_{var_name}'] = df[f'REC_{var_name}'].mask(mask)

    observed_df = df[[col for col in df.columns if col.startswith("OBS")]]
    predictions_df = df[[col for col in df.columns if col.startswith("REC")]]
    predictions_df.columns = observed_df.columns
    return observed_df, predictions_df

def get_rp(ldt, lt, st):
    rp = {}
    rp['Tmax'] = ldt.TIME.max()

    rp["static_vnames"] = st["Variable"].dropna().unique().tolist()
    rp["static_cat"] = st[st["Type"] == "cat"]["Variable"].dropna().unique().tolist()
    rp["static_cont"] = st[st["Type"] != "cat"]["Variable"].dropna().unique().tolist()

    rp["long_vnames"] = lt["Variable"].dropna().unique().tolist()
    rp["long_cat"] = lt[lt["Type"] == "cat"]["Variable"].dropna().unique().tolist()
    rp["long_bin"] = lt[(lt["Type"] == "cat") & (lt["Cats"] == 2)]["Variable"].dropna().unique().tolist()
    rp["long_cont"] = lt[lt["Type"] != "cat"]["Variable"].dropna().unique().tolist()

    return rp

def convert_long_data(df0, only_pos=False):

    df1 = df0.melt(id_vars=["PTNO", "REPI", "TIME", "DRUG"], 
                   var_name="FullVar", value_name="DV")
    
    df1[['TYPE', 'Variable']] = df1['FullVar'].str.extract(r'([^_]+)_(.*)')
    df1.drop(columns='FullVar', inplace=True)
    df1["TIME2"] = (df1["TIME"] * 1_000_000).round().astype(int)
    if only_pos:
        df1["DV"] = df1["DV"].clip(lower=0)
    df_main = df1[df1["TYPE"] != "MASK"].copy()
    df_mask = df1[df1["TYPE"] == "MASK"].copy()

    df_mask = df_mask.drop(columns=["TIME", "TYPE", "DRUG"]).rename(columns={"DV": "MASK"})
    df_final = df_main.merge(df_mask, on=["PTNO", "REPI", "Variable", "TIME2"], how="left")
    df_final["TIME"] = df_final["TIME"].round(2)
    df_final.drop(columns="TIME2", inplace=True)
    df_final.rename(columns={"PTNO": "SUBJID"}, inplace=True)
    return df_final

def convert_static_data(df0, only_pos=False):
    df1 = df0.melt(id_vars=["PTNO", "REPI"], 
                   var_name="FullVar", 
                   value_name="DV")
    df1[["TYPE", "Variable"]] = df1["FullVar"].str.extract(r"^(.*)_(.*)$")
    df1.drop(columns='FullVar', inplace=True)
    if only_pos:
        df1["DV"] = df1["DV"].clip(lower=0)
    df_mask = df1[df1["TYPE"] == "MASK"].drop(columns="TYPE").rename(columns={"DV": "MASK"})
    df_final = df1[df1["TYPE"] != "MASK"].merge(df_mask, on=["PTNO", "REPI", "Variable"], how="left")
    df_final = df_final.rename(columns={"PTNO": "SUBJID"})
    return df_final

def convert_data(df0,type,only_pos=False):
    if type=='long':
        df_final = convert_long_data(df0,only_pos=only_pos)
    else:
        df_final = convert_static_data(df0,only_pos=only_pos)

    df_final = df_final[
        ((df_final["TYPE"] == "OBS") & (df_final["MASK"] == 1)) | (df_final["TYPE"] != "OBS")]

    df_final = df_final.drop(columns=["MASK"])

    df_final["TYPE"] = df_final["TYPE"].replace({
        "OBS": "Observed",
        "REC": "Reconstructed",
        "SIM": "Simulations"})
    return df_final