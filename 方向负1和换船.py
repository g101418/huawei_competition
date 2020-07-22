
# 方向问题
train_data_ = train_data[(train_data['direction']>=0)&(train_data['direction']<=36000)].reset_index(drop=True)


# 换船问题

def ff(x):
    mssi = x.vesselMMSI.unique().tolist()
    if len(mssi) > 1:
        mssi_lengths = []
        
        for ms in mssi:
            ms_df = x[x['vesselMMSI']==ms]
            mssi_lengths.append([ms, len(ms_df)])
        
        mssi_lengths.sort(key=lambda x:x[1], reverse=True)
        
        max_mssi = mssi_lengths[0][0]
        
        ms_df_not_index_list = x[x['vesselMMSI']!=ms].index.tolist()
        
        return ms_df_not_index_list
    
    
    return []
    
aa=train_data_.groupby('loadingOrder').parallel_apply(ff)


bb=aa.tolist()
bb=[j for i in bb for j in i]


train_data_ = train_data_.drop(labels=bb, axis=0)
train_data_ = train_data_.sort_values(['loadingOrder', 'timestamp'])
train_data_ = train_data_.reset_index(drop=True)