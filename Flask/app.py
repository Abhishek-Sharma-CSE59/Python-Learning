from flask import Flask , request,render_template

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    if request.method =="POST":
        user_input = request.form.get("user_input")     #get data from the form
        if user_input:
            processed_data = process_input(user_input)  #process the data
            return render_template("index.html", result = processed_data, input= user_input)
        else:
            return render_template("index.html",error = "Please enter some text.", input="")
    return render_template("index.html",input="")

def process_input(text):
    return text.upper()[::-1]


if __name__ =="__main__":
    app.run(debug=True)

