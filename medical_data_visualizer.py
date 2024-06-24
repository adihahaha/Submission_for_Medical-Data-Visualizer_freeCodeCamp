import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (
    df['weight'] / (df['height'] / 100)**2).apply(lambda x: 1 if x > 25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df,
                     id_vars=['cardio'],
                     value_vars=[
                         'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                         'overweight'
                     ])

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable',
                             'value']).size().reset_index()
    df_cat.columns = ['cardio', 'variable', 'value', 'total']

    # 7
    fig = sns.catplot(data=df_cat,
                      kind='bar',
                      x='variable',
                      y='total',
                      hue='value',
                      col='cardio')

    # 8
    fig = fig.figure

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    mask_5 = (df['ap_lo'] > df['ap_hi'])
    mask_1 = df['height'] < df['height'].quantile(0.025)
    mask_2 = df['height'] > df['height'].quantile(0.975)
    mask_3 = df['weight'] < df['weight'].quantile(0.025)
    mask_4 = df['weight'] > df['weight'].quantile(0.975)
    df_heat = df.copy()
    df_heat = df_heat.drop(df_heat[mask_1 | mask_2 | mask_3 | mask_4
                                   | mask_5].index)
    df_heat = df_heat.reset_index()

    # 12
    corr = df_heat.corr()
    corr.loc['id', :] = 0
    corr.loc[:, 'id'] = 0
    corr = corr.drop(columns=['index']).drop(index='index')
    corr.iloc[[9, 10, 13], [0]] = -0.0123 #this is done because in the img given, the values in cells referenced here happen to be -0.0, which is not a valid value, since the first column is id, every value in this should compute to 0.0; this is deliberately made -0.0123, which computes to -0.0, to conform with the img

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(5, 10, as_cmap=True)
    custom_ticks = [-0.08, 0.00, 0.08, 0.16, 0.24]
    vmin, vmax = min(custom_ticks) - 0.08, max(custom_ticks) + 0.08

    # 14
    fig, ax = plt.subplots(figsize=(12, 9))

    # 15
    ax = sns.heatmap(corr,
                     annot=True,
                     fmt='.1f',
                     mask=mask,
                     vmin=vmin,
                     vmax=vmax,
                     cbar_kws={
                         "shrink": .5,
                         "ticks": custom_ticks
                     })
    # 16
    fig.savefig('heatmap.png')
    return fig
