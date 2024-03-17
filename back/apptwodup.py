from flask import Flask, jsonify, request, render_template,session
from flask_cors import CORS
import yfinance as yf
from datetime import date, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from flask_bcrypt import Bcrypt
from flask_session import Session
from config import ApplicationConfig
from models import db, User



app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config.from_object(ApplicationConfig)
db.init_app(app)
bcrypt = Bcrypt(app)
Session(app)

@app.route("/@me")
def get_current_user():
    user_id = session.get("user_id")

    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    user = User.query.filter_by(id=user_id).first()
    return jsonify({
        "id": user.id,
        "email": user.email
    }) 

@app.route("/register", methods=["POST"])
def register_user():
    data = request.get_json()
    email = data["email"]
    password = data["password"]

    user_exists = User.query.filter_by(email=email).first()

    if user_exists:
        return jsonify({"error": "User already exists"}), 409

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    session["user_id"] = new_user.id

    return jsonify({
        "id": new_user.id,
        "email": new_user.email
    })

@app.route("/login", methods=["POST"])
def login_user():
    data = request.get_json()
    email = data["email"]
    password = data["password"]

    user = User.query.filter_by(email=email).first()

    if user is None:
        return jsonify({"error": "Login details are incorrect"}), 401

    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Unauthorized"}), 401
    
    session["user_id"] = user.id

    return jsonify({
        "id": user.id,
        "email": user.email
    })

@app.route("/logout", methods=["POST"])
def logout_user():
    session.pop("user_id", None)
    return jsonify({"message": "Successfully logged out"})

@app.route('/predict_stock', methods=['POST'])
def predict_stock():
    data = request.get_json()
    selected_stock = data['selected_stock']
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d") 
    data = yf.download(selected_stock, START, TODAY)
    data = yf.download(selected_stock, START, TODAY)
    data = yf.download(selected_stock, START, TODAY)
    data.reset_index(inplace=True)

    raw_data = data.to_dict(orient='records')

    # Data Preprocessing
    missing_values_count = data.isnull().sum()
    missing_values_before_str = missing_values_count.to_string()

    if missing_values_count.sum() > 0:
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        data[numerical_cols] = data[numerical_cols].fillna(data.mean())

    columns_to_drop = ['Volume', 'Adj Close']
    data.drop(columns_to_drop, inplace=True, axis=1)
    
    preprocessing_info = f"Missing values count:\n{missing_values_before_str}\nDropped columns: {', '.join(columns_to_drop)}"

    if data['Close'].std() == 0 or data['High'].std() == 0:
        pearsons_correlation = 'Undefined (no variance in data)'
    else:
        correlation = data.corr()
        pearsons_correlation = correlation.at['Close', 'High']
        if np.isnan(pearsons_correlation):
            pearsons_correlation_str = "NaN (not computable)"
        else:
            pearsons_correlation_str = f"{pearsons_correlation:.2f}"
    
    min_high = data['High'].min()
    min_close = data.loc[data['High'].idxmin()]['Close']
    max_high = data['High'].max()
    max_close = data.loc[data['High'].idxmax()]['Close']

    close_values = data['Close'].tolist()
    high_values = data['High'].tolist()


    X_train, X_test, Y_train, Y_test = train_test_split(data[['Open', 'High', 'Low']], data['Close'], test_size=0.3, random_state=40)

    svt_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svt_rbf.fit(X_train, Y_train)
    y_rbf = svt_rbf.predict(X_test)

    reg_lasso = linear_model.Lasso()
    reg_lasso.fit(X_train, Y_train)
    prd_lasso = reg_lasso.predict(X_test)

    score_svr_mae = metrics.mean_absolute_error(Y_test, y_rbf)
    score_svr_mse = metrics.mean_squared_error(Y_test, y_rbf)
    score_svr_rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_rbf))

    score_lasso_mae = metrics.mean_absolute_error(Y_test, prd_lasso)
    score_lasso_mse = metrics.mean_squared_error(Y_test, prd_lasso)
    score_lasso_rmse = np.sqrt(metrics.mean_squared_error(Y_test, prd_lasso))

    future_dates = pd.date_range(data['Date'].iloc[-1] + timedelta(days=1), periods=10, freq='D')
    predictions = []

    present_date = pd.Timestamp(date.today())

    for i in range(10):
        if future_dates[i] > present_date:
            predictions.append({
                'date': future_dates[i].strftime('%d-%b-%Y'),
                'price': prd_lasso[i]
            })

    response = jsonify({
        'preprocessingInfo': preprocessing_info,
        'correlation': f"Pearson's correlation: {pearsons_correlation:.2f}",
        'closeValues': close_values,
        'highValues': high_values,
        'minHigh': min_high,
        'minClose': min_close,
        'maxHigh': max_high,
        'maxClose': max_close,
        'totalData': len(data),
        'trainData': len(X_train),
        'testData': len(X_test),
        'predictions': predictions,
        'svr_metrics': {
            'mae': score_svr_mae,
            'mse': score_svr_mse,
            'rmse': score_svr_rmse
        },
        'lasso_metrics': {
            'mae': score_lasso_mae,
            'mse': score_lasso_mse,
            'rmse': score_lasso_rmse
        },
        'rawData': raw_data
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    # response.headers.add('Access-Control-Allow-Credentials', 'true')

    return response

if __name__ == '__main__':
    app.run(debug=True)
