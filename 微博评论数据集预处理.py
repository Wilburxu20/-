import re
import pandas as pd

def preprocess_weibo(text):
    # 删除//后面的回复内容
    text = re.sub(r'//', '', text)
    # 删除@后面的昵称，但保留@符号后的其他内容
    text = re.sub(r'@\w+[:：]', '', text)
    # 去除URL、@用户、话题标签等
    text = re.sub(r'http\S+|@\S+|#\S+#', '', text)
    # 删除所有【】及其内容
    text = re.sub(r'【.*?】', '', text)
    # 删除"回复"二字
    text = re.sub(r'回复[:：]\s*', '', text)
    # 删除多余的空白字符并返回
    return ' '.join(text.split()).strip()

# 读取数据
data = pd.read_csv('data0411.csv')

# 应用预处理
data['processed_review'] = data['comment'].apply(preprocess_weibo)
data.drop(columns='comment', inplace=True)

# 删除 NaN 和空字符串
data.dropna(inplace=True)
data.replace("", pd.NA, inplace=True)
data.dropna(inplace=True)

# 删除含有广告句子的评论
ad_keywords = '|'.join(['抢购', '优惠', '折扣', '促销', '团购', '秒杀', '下单', '购买', '淘宝', '京东', '天猫', '微店', '官网', '回复', '顾问'])
df_filtered = data[~data['processed_review'].str.contains(ad_keywords, na=False)]

# 筛选出'processed_review'列中长度大于10的数据
df_final = df_filtered[df_filtered['processed_review'].str.len() > 10]

# 更改列名
df_final.rename(columns={'processed_review': 'comment'}, inplace=True)

pattern = r'[０-９0-9Ａ-Ｚａ-ｚA-Za-z]'

# 过滤包含数字或字母的评论行
cleaned_df = df_final[~df_final['comment'].str.contains(pattern, na=False, regex=True)]

sample_0 = cleaned_df[cleaned_df['label'] == 0].sample(n=5000, random_state=42)
sample_1 = cleaned_df[cleaned_df['label'] == 1].sample(n=5000, random_state=42)

# 合并并打乱顺序
combined = pd.concat([sample_0, sample_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# 检查结果
print(combined['label'].value_counts())

# 检查处理后的数据
print(combined.isna().sum())

# 打印初始和处理后的数据长度
print(f'Original data length: {len(data)}')
print(f'Final data length: {len(combined)}')

# 保存处理后的数据
combined.to_csv('data_final.csv', index=False)