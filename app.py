import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show TensorFlow errors

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from predict import predict_image  # Your image analysis function

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    # Create a simple static blog post
    blog_post = {
        'title': 'Welcome to NIDAN AI Blog',
        'content': 'Stay tuned for updates on our medical image analysis technology. We are working on bringing you the latest developments in AI-powered medical image analysis, research insights, and healthcare technology innovations.',
        'image': '/static/src/blog-placeholder.jpg',
        'category': 'News',
    }
    return render_template('blog.html', article=blog_post)

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('upload.html', message="No file selected")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction, segmented_path = predict_image(filepath)

        return render_template('upload.html', filename=filename, prediction=prediction,
                               segmentation_path=segmented_path)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)