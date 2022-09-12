# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
#
# document = 'diversityST'
# sub_document = 'COVS'  # '' #'topic' COVS AREL
#
# df = pd.read_csv('/home/frc/code/vistcode/vist_eval/{}/{}/{}_word.csv'.format(document,sub_document,sub_document)
#
# plt.figure(figsize=(16,10), dpi= 80)
# sns.kdeplot(df.loc[df['cyl'] == 4, "cty"], shade=True, color="g", label="Cyl=4", alpha=.7)
#
# plt.title('Density Plot of City Mileage by n_Cylinders', fontsize=22)
# plt.legend()
# plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



# Import Data
df = pd.read_csv("C:\Users\\10632\Desktop\diversity.csv")

# Draw Plot
plt.figure(figsize=(16, 10), dpi=80)
sns.kdeplot(df.loc[df['cyl'] == 4, "cty"], shade=True, color="g", label="Cyl=4", alpha=.7)
sns.kdeplot(df.loc[df['cyl'] == 5, "cty"], shade=True, color="black", label="Cyl=5", alpha=.7)
sns.kdeplot(df.loc[df['cyl'] == 6, "cty"], shade=True, color="dodgerblue", label="Cyl=6", alpha=.7)
# sns.kdeplot(df.loc[df['cyl'] == 8, "cty"], shade=True, color="orange", label="Cyl=8", alpha=.7)

# Decoration
plt.title('Density Plot of City Mileage by n_Cylinders', fontsize=22)
plt.legend()

