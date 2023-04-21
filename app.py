from flask import Flask,render_template,request,redirect,jsonify
import torch
import pathlib
from datetime import datetime as date
import os
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import glob
import wikipedia

# device = 0 if torch.cuda.is_available() else -1
# print('load model...')
# summarizer_bart_large_cnn = pipeline("summarization", model="facebook/bart-large-cnn",device=device)
# print('load model complete')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)


app = Flask(__name__)
def fileType(filename):
    filename = str(filename)
    temp = filename.split('.')
    return temp[-1]
def pdfReader(path):
    text = []
    pdfFileObj = open(path, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    for page in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(page)
        text.append(pageObj.extractText())
    pdfFileObj.close()
    text = ' '.join(text)
    return text
def summaryProcess(articles):
    summarys = []
    for article in articles:
        tokens_input = tokenizer.encode("summarize: "+article, return_tensors='pt', max_length=512, truncation=True).to(device)
        ids = model.generate(tokens_input, min_length=80, max_length=120)
        summary = tokenizer.decode(ids[0], skip_special_tokens=True)
        summarys.append(summary)
    return summarys
def summaryWiki(search):
    summary = ''
    title = ''
    try:
        page = wikipedia.page(search,auto_suggest=False)
        article = page.content
        title = page.title
        url = page.url
        tokens_input = tokenizer.encode("summarize: "+article, return_tensors='pt', max_length=512, truncation=True).to(device)
        ids = model.generate(tokens_input, min_length=80, max_length=120)
        summary = tokenizer.decode(ids[0], skip_special_tokens=True)
        #return (title,summary,url,page.images)
        return (title,summary,url,page.images)
    except:
        #return (None,None,None,None)
        return (None,None,None)

def searchWiki(search):
    try:
        page = wikipedia.page(search,auto_suggest=False)
        return page.title
    except:
        return ''


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summary", methods=['POST'])
def summary():
    currentTime = str(date.now()).replace('.','_').replace(':','-')
    articles = request.form.getlist('articles')
    filename = request.form.get('filename')
    time = request.form.get('time')
    #print(filename)
    if(len(articles) != 0):#input with type
        filenames = ['']
        if(len(articles[0]) > 0):
            summarized = summaryProcess(articles)
            results = list(zip(filenames,summarized))
            return render_template("summary.html",results=results)
        return redirect("/")
    elif(filename and time):#input with myfiles
        filenames = [filename]
        filename = time+'_'+filename 
        path = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/text_files/{filename}"
        txt = ''
        if(fileType(filename) == 'txt'):
            f = open(path, 'r',encoding="utf-8")
            txt = f.read()
            f.close()
        elif(fileType(filename) == 'pdf'):
            txt = pdfReader(path)
        else:
            return redirect("/myfiles")     
        articles = [txt] 
        summarized = summaryProcess(articles)
        results = list(zip(filenames,summarized))
        return render_template("summary.html",results=results)
    else:#input with files
        articles = []
        filenames = []
        if 'file[]' not in request.files:
            #flash('No file part')
            return redirect("/")
        files = request.files.getlist("file[]")
        for file in files:
            if file.filename == '':
                return redirect("/")
            if file:
                filename = file.filename
                filenames.append(filename)
                fileName_toSave = currentTime+'_'+filename
                path = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/text_files/{fileName_toSave}"
                file.save(path)
                if(fileType(filename) == 'txt'):
                    #print('txt')
                    f = open(path, 'r',encoding="utf-8")
                    txt = f.read()
                    f.close()
                    articles.append(txt)
                elif(fileType(filename) == 'pdf'):
                    #print('pdf')
                    txt = pdfReader(path)
                    articles.append(txt)
                else:
                    return redirect("/")

        summarized = summaryProcess(articles)
        results = list(zip(filenames,summarized))
        return render_template("summary.html",results=results)

@app.route("/myfiles")
def myfiles():
    FileTypes = ('*.txt','*.pdf')
    filenames = []
    times = []
    all_files = []
    for filestype in FileTypes:
        path = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/text_files/{filestype}"
        all_files += glob.glob(path)
    all_files.sort(reverse = True)
    for filePath in all_files:
        fileName = filePath.split('\\')
        fileName = fileName[len(fileName)-1]
        fileName = fileName.split("_")
        if(len(fileName) < 3):
            continue
        time = fileName[0]+"_"+fileName[1]
        times.append(time)
        fileName = fileName[2:]
        fileName = '_'.join(fileName)
        filenames.append(fileName)

    #filenames = enumerate(filenames)
    data = {'filenames':filenames,'times':times}
    return render_template('myfiles.html',data=data)

@app.route('/getfile_content', methods=['POST'])
def getfile_content():
    filename = request.json['filename']
    time = request.json['time']
    realName = time+'_'+filename
    path = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/text_files/{realName}"
    txt = ''
    if(fileType(filename) == 'txt'):
        f = open(path, 'r',encoding="utf-8")
        txt = f.read()
        f.close()
    elif(fileType(filename) == 'pdf'):
        txt = pdfReader(path)
    return jsonify({'message':'success','fileContent':txt})

@app.route('/delfile', methods=['POST'])
def delfile():
    filename = request.json['filename']
    time = request.json['time']
    realName = time+'_'+filename
    path = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/text_files/{realName}"
    if os.path.exists(path):
        os.remove(path)
        return jsonify({'message':'success'})
    else:
        return jsonify({'message':'fail'})

@app.route('/wikipedia')
def wiki():
    return render_template('wikipedia.html')
    
@app.route('/searchwiki', methods=['POST'])
def searchwiki():
    search = request.json['search']
    title = searchWiki(search)
    return jsonify({'message':'success','title':title,'search':search})

@app.route('/wikiresult', methods=['POST'])
def wikipedia_search():
    search = request.form.get('search')
    title,summary,url,img = summaryWiki(search)
    #return render_template('wikiresult.html',summary=summary,title=title,search=search)
    return render_template('wikiresult.html',search=search,title=title,summary=summary,url=url,img=img)

if __name__ == '__main__':
    app.run(debug=True)