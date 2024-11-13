#price forecasting
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#read the dataset
dataset=pd.read_excel(r'C:\Users\nahas\phd_interview\Final.xlsx', sheet_name='Dataset')
dataset.replace('n/e', 0, inplace=True)
df=dataset.copy()


# Extract the start time part from the 'MTU' column
df['MTU_cleaned'] = df['MTU'].str.split(' - ').str[0]

# Convert the cleaned 'MTU_cleaned' column to datetime
df['MTU'] = pd.to_datetime(df['MTU_cleaned'], format='%d.%m.%Y %H:%M')

# Drop the helper column if you don't need it anymore
df.drop(columns='MTU_cleaned', inplace=True)
#df = df[df['MTU'].dt.year != 2022]


df = df.set_index('MTU')
df.info()
#####################################################################################
###########################Preprocessing step##########################################
# Find NaNs and duplicates in df
print('There are {} missing values or NaNs in df.'
      .format(df.isnull().values.sum()))

#Duplicated values
temp_energy = df.duplicated(keep='first').sum()
print('There are {} duplicate rows in df based on all columns.'
      .format(temp_energy))
df = df.drop_duplicates()
df.info()

#Find the number of NaNs in each column
df.isnull().sum(axis=0)
# Fill null values using interpolation
df.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)
##############################################################################
#################### plot of all features vs target column##############################
############################################################################
features = df.columns.drop('Target_price_NO1')  # List of features excluding the target column

# Create a 6x4 grid for subplots (6 rows, 4 columns)
fig = make_subplots(rows=6, cols=4, subplot_titles=features)

# Loop through each feature and plot it against the target variable
for i, feature in enumerate(features):
    row = (i // 4) + 1  # Calculate row number (starts at 1)
    col = (i % 4) + 1   # Calculate column number (starts at 1)
    
    fig.add_trace(
        go.Scatter(
            x=df[feature],
            y=df['Target_price_NO1'],
            mode='markers',
            name=feature,
            line=dict(width=2)
        ),
        row=row, col=col
    )

# Update layout: set the title and adjust layout
fig.update_layout(
    height=1200, width=1400,  # Adjust size for better visualization
    title_text="Feature vs Target Price Subplots",
    showlegend=False
)

# Show the plot
fig.show()
#plt.subplot(2,2,1)
#plt.title('Hydro Run-of-river_NO1 VS Target_price_NO1')
#plt.scatter(df['Hydro Run-of-river_NO1'],df['Target_price_NO1'],s=2, c='g')
##############################################################################
#################### plot of just the target column##############################
############################################################################
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,  # Use the DataFrame's index for the x-axis (or another time-based column if applicable)
    y=df['Target_price_NO1'],  # The target column you want to plot
    mode='lines',  # Line plot
    name='Target Price NO1',
    line=dict(color='blue', width=2)  # Customize line color and width
))

# Set the title and axis labels
fig.update_layout(
    title='Target Price NO1 Over Time',
    xaxis_title='Index',
    yaxis_title='Target Price NO1'
)

# Show the figure
fig.show()
##############################################################################
#########df['Actual Total Load_NO1']-df['Hydro Run-of-river_NO1']-df['Wind Onshore_NO1']#############n                        ##############################
#############################################################################
new_column=df['Actual Total Load_NO1']-df['Hydro Run-of-river_NO1']-df['Wind Onshore_NO1']
fig = go.Figure()

fig.add_trace(go.Scatter(
    y=df['Target_price_NO1'],  # Use the DataFrame's index for the x-axis (or another time-based column if applicable)
    x=new_column,  # The target column you want to plot
    mode='markers',  # Line plot
    name='Target Price NO1',
    line=dict(color='blue', width=2)  # Customize line color and width
))

# Set the title and axis labels
fig.update_layout(
    title='Target Price NO1 Over Time',
    xaxis_title='Index',
    yaxis_title='Target Price NO1'
)

# Show the figure
fig.show()
##############################################################################
####################  Correlation        ##############################
############################################################################
correlation= df.corr()
##############################################################################
####################Normalization            ##############################
############################################################################
scaler = MinMaxScaler()

scaled_df = scaler.fit_transform(df)
df = pd.DataFrame(scaled_df)

data = df.values
# ##############################################################################
####################Train and test split              ##############################
############################################################################
X = data[:, :-1]
Y = data[:, -1]

x_train = X[0:26000]
x_valid = X[26001:29500]
x_test = X[29500:]

y_train = Y[0:26000]
y_valid = Y[26001:29500]
y_test = Y[29500:]


past = 240#for 10 days
learning_rate = 0.001
batch_size = 256
epochs = 10
step=1
sequence_length = int(past / step)
dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_valid,
    y_valid,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)
# ##############################################################################
####################Training model            ##############################
############################################################################
for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(256)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

# ##############################################################################
####################fit model            ##############################
############################################################################
path_checkpoint = "model_checkpoint.weights.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)
########################################################################################
#######################           plot loss           ################################
########################################################################################
def visualize_loss_plotly(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))

    # Create the plot
    fig = go.Figure()

    # Add Training loss
    fig.add_trace(go.Scatter(
        x=list(epochs), y=loss,
        mode='lines',
        name='Training loss',
        line=dict(color='blue')
    ))

    # Add Validation loss
    fig.add_trace(go.Scatter(
        x=list(epochs), y=val_loss,
        mode='lines',
        name='Validation loss',
        line=dict(color='red')
    ))

    # Set the title and axis labels
    fig.update_layout(
        title=title,
        xaxis_title='Epochs',
        yaxis_title='Loss',
        legend_title='Loss Type'
    )

    # Show the plot
    fig.show()

# Call the function
visualize_loss_plotly(history, "Training and Validation Loss")
########################################################################################
#######################           prediction and metrics          ################################
########################################################################################
dataset_test = keras.preprocessing.timeseries_dataset_from_array(
    x_test,
    None,  # You can pass None here if you're only predicting and don't need the targets
    sequence_length=sequence_length,  # Ensure it's the same as during training
    sampling_rate=step)

y_pred=model.predict(dataset_test)
y_pred = y_pred.flatten()

mse = mean_squared_error(y_test[-len(y_pred):], y_pred)
mae = mean_absolute_error(y_test[-len(y_pred):], y_pred)
r2 = r2_score(y_test[-len(y_pred):], y_pred)

print(f'MSE is: {mse}')
print(f'MAE is: {mae}')
print(f'R-squared is: {r2}')

########################################################################################
#######################           plot Prediction vs actual    ################################
########################################################################################
y_test_aligned = y_test[-len(y_pred):]  # Adjust y_test to match the length of y_pred if needed

# Create a Plotly figure
fig = go.Figure()

# Plot Actual Prices
fig.add_trace(go.Scatter(
    x=list(range(len(y_test_aligned))),  # x-axis can be time steps or indices
    y=y_test_aligned,  # Actual values
    mode='lines',
    name='Actual Prices',
    line=dict(color='blue')
))

# Plot Predicted Prices
fig.add_trace(go.Scatter(
    x=list(range(len(y_pred))),  # x-axis can be time steps or indices
    y=y_pred,  # Predicted values
    mode='lines',
    name='Predicted Prices',
    line=dict(color='red', dash='dash')  # Dash style for differentiation
))

# Customize the layout
fig.update_layout(
    title='Actual vs Predicted Prices',
    xaxis_title='Time Steps',
    yaxis_title='Price',
    legend_title='Legend',
    height=600,
    width=1000
)

# Show the plot
fig.show()





