## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('data.csv')
print(df)
```
<img width="627" height="276" alt="Screenshot 2026-05-20 183008" src="https://github.com/user-attachments/assets/fddad125-8938-45fb-84de-7ad1d107e336" />

```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
climate = ['Cold','Warm','Hot','Very Hot']
ele = OrdinalEncoder(categories=[climate])
ele.fit_transform(df[["Ord_1"]])
```

<img width="412" height="251" alt="Screenshot 2026-05-20 183014" src="https://github.com/user-attachments/assets/ee407082-aa55-4871-b013-378a2038e4d0" />
```
df['bo2'] = ele.fit_transform(df[["Ord_1"]])
df
```

<img width="681" height="372" alt="Screenshot 2026-05-20 183021" src="https://github.com/user-attachments/assets/15d875bc-af6e-4f28-bd37-b4243d231595" />

```
le = LabelEncoder()
df2 = df.copy()
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
```

<img width="687" height="380" alt="Screenshot 2026-05-20 183027" src="https://github.com/user-attachments/assets/7cfc32e5-4309-41a1-a6da-1d5a9f3bc02b" />
```
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
```

<img width="675" height="401" alt="Screenshot 2026-05-20 183033" src="https://github.com/user-attachments/assets/1a24262a-13b9-475a-ae53-e5c48a0f1c79" />

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
df3 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["City"]]))
df2 = pd.concat([enc,df3],axis = 1)
df2
```

<img width="722" height="408" alt="Screenshot 2026-05-20 183040" src="https://github.com/user-attachments/assets/820a624a-53db-4e8f-b7b7-a23a4d2cb51d" />

```
pd.get_dummies(df,columns=['City'])
```

<img width="1058" height="406" alt="Screenshot 2026-05-20 183048" src="https://github.com/user-attachments/assets/8dc9ef1b-1113-4870-a02d-59e5230743cc" />

```
from category_encoders import BinaryEncoder
df = pd.read_csv('data.csv')
df
```

<img width="746" height="400" alt="Screenshot 2026-05-20 183054" src="https://github.com/user-attachments/assets/a6143639-3c95-4ae7-9758-65af359cd2c8" />

```
be = BinaryEncoder()
nd = be.fit_transform(df['Ord_2'])
df
```

<img width="641" height="412" alt="Screenshot 2026-05-20 183100" src="https://github.com/user-attachments/assets/44683ec9-0c97-484a-8b94-1d7d8bc959a5" />

```
from category_encoders import TargetEncoder
te = TargetEncoder()
CC = df.copy()
new = te.fit_transform(CC["City"],y=CC["Target"])
CC = pd.concat([CC,new],axis = 1)
CC
```

<img width="765" height="387" alt="Screenshot 2026-05-20 183106" src="https://github.com/user-attachments/assets/28350973-4bb7-4961-867a-edbe8aa9d6de" />

```
if 'City' in CC.columns:
    CC = CC.drop('City', axis=1)
new = te.fit_transform(X = df["City"],y=df["Target"])
CC = pd.concat([CC.reset_index(drop=True),new.reset_index(drop=True)],axis = 1)
CC
```

<img width="651" height="403" alt="Screenshot 2026-05-20 183111" src="https://github.com/user-attachments/assets/6ded9767-76c3-4d21-a1ae-464026ec7d75" />

```
df = pd.read_csv('Data_to_Transform.csv')
df
```

<img width="942" height="470" alt="Screenshot 2026-05-20 183118" src="https://github.com/user-attachments/assets/7d55c017-9a01-4f36-8207-e409b4da299b" />

```
df.skew()
```

<img width="542" height="120" alt="Screenshot 2026-05-20 183125" src="https://github.com/user-attachments/assets/ceff6d57-688f-4911-b251-619dfd768dd2" />

```
np.log(df["Highly Positive Skew"])
```

<img width="698" height="265" alt="Screenshot 2026-05-20 183130" src="https://github.com/user-attachments/assets/176fe1cc-2f2b-4b78-a110-57f16efb7dbe" />

```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="741" height="277" alt="Screenshot 2026-05-20 183135" src="https://github.com/user-attachments/assets/707f1bab-cdbc-45fc-a73a-ff3505ab8f89" />

```
np.sqrt(df["Highly Positive Skew"])
```

<img width="730" height="307" alt="Screenshot 2026-05-20 183141" src="https://github.com/user-attachments/assets/815db421-6ae3-4cea-b48a-37da6ea84367" />

```
np.square(df["Highly Positive Skew"])
```

<img width="762" height="276" alt="Screenshot 2026-05-20 183148" src="https://github.com/user-attachments/assets/53682b49-aad8-48a4-919d-ccb8dc584d39" />

```
df["Highly Positive Skew_boxcox"], parameters = stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="1157" height="467" alt="Screenshot 2026-05-20 183200" src="https://github.com/user-attachments/assets/897334c5-5f1f-4db8-a14c-d73aeabfafff" />

```
df["Moderate Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Moderate Negative Skew"])
df
```
<img width="1357" height="480" alt="Screenshot 2026-05-20 183225" src="https://github.com/user-attachments/assets/99bf66e1-6a0a-4708-ab4d-64b4a9172f06" />

```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="787" height="536" alt="Screenshot 2026-05-20 183353" src="https://github.com/user-attachments/assets/07fe1442-5bf7-45f0-b988-f79d14a7ae65" />

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line = '45')
plt.show()
```

<img width="943" height="513" alt="Screenshot 2026-05-20 183400" src="https://github.com/user-attachments/assets/5da66c1f-32c7-4344-92e1-b03b6f5abf5d" />

```
sm.qqplot(df["Moderate Negative Skew_1"],line = '45')
plt.show()
```

<img width="802" height="542" alt="Screenshot 2026-05-20 183408" src="https://github.com/user-attachments/assets/b6dc87fe-ec26-4a53-b7ba-8f469380aa92" />

```
df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line = '45')
plt.show()
```

<img width="816" height="541" alt="Screenshot 2026-05-20 183644" src="https://github.com/user-attachments/assets/7e3a4319-cedd-4b7b-9dfd-90f03c7d1458" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line = '45')
plt.show()
```

<img width="857" height="540" alt="Screenshot 2026-05-20 183651" src="https://github.com/user-attachments/assets/98f8d2fe-f0c4-497d-a751-df805b1bbb88" />

```
sm.qqplot(df["Highly Negative Skew_1"],line = '45')
plt.show()
```

<img width="800" height="533" alt="Screenshot 2026-05-20 183752" src="https://github.com/user-attachments/assets/9d46086f-0e6c-44bb-be51-8b09d6470877" />

```
sm.qqplot(np.abs(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```

<img width="838" height="570" alt="Screenshot 2026-05-20 183801" src="https://github.com/user-attachments/assets/9ee32496-a8b5-4931-b14e-bd9977ee9d44" />

```
sm.qqplot(np.log(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```

<img width="767" height="531" alt="Screenshot 2026-05-20 183807" src="https://github.com/user-attachments/assets/8bbab9eb-5417-4126-b515-dcf7be87e61d" />

```
sm.qqplot(np.sqrt(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```

<img width="725" height="415" alt="Screenshot 2026-05-20 183813" src="https://github.com/user-attachments/assets/f2d248ef-e7fc-43bf-9a91-c34356025ac5" />


# RESULT:
       # INCLUDE YOUR RESULT HERE

       
