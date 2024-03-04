from flask import Flask , render_template , request
import pickle


model = pickle.load(open("GB.pkl" , 'rb'))
pipe = pickle.load(open("pipe.pkl" , 'rb'))

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/predict" , methods = ['POST'])
def predict():
    crop = request.form['crop']
    state = request.form['state']
    a2fl = request.form['a2fl']
    c2 = request.form['c2']
    c2p = request.form['c2p']

    answer = model.predict(pipe.transform([[crop.upper() , state.capitalize() , a2fl , c2 , c2p]]))

    return render_template("index.html" , result = round(answer[0] , 2))


if __name__ == "__main__":
    app.run(debug = True)