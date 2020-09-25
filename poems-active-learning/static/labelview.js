const getJSON = function (url) {
    return new Promise((res, rej) => {
        let xhr = new XMLHttpRequest();
        xhr.open('GET', url, true);
        xhr.responseType = 'json';
        xhr.onload = function () {
            const status = xhr.status;
            if (status === 200) {
                res(xhr.response);
            } else {
                rej(xhr.response);
            }
        };
        xhr.send();
    });
};

const app = {};

app.processLabelBatch = function(data) {
    app.documents = data.docs;
    app.documents.forEach(d => { d.labels = {} });

    document.querySelector('.label-overview').innerHTML = '<span></span>'.repeat(data.docs.length);
    document.querySelector('#date').innerHTML = new Date(data.date).toLocaleString();

    app.focusDoc(0)

};

app.focusDoc = function(index) {
    window.scrollTo(0, 0)
    document.querySelectorAll('.label-overview > span').forEach((el, i) => {
        if (i == index) el.classList.add('focused');
        else el.classList.remove('focused');
    });

    app.curindex = index;
    const doc = app.documents[index];
    document.querySelector('#docid').innerHTML = `${index+1}/${app.documents.length}`;
    document.querySelector('#filename').innerHTML = doc.filename;
    document.querySelector('#prediction').innerHTML = doc['predicted_labels'];
    document.querySelector('#doc-content').innerHTML = doc.content;

    document.querySelectorAll('.label-button').forEach((el) => {
        if (doc.labels[el.attributes['data-label'].value]) {
            el.classList.add('selected');
        } else {
            el.classList.remove('selected');
        }
    });
};

app.setLabel = function(label, val) {
    const doc = app.documents[app.curindex];
    doc.labels[label] = val;

    const indicator = document.querySelectorAll('.label-overview span')[app.curindex];
    if (Object.values(doc.labels).reduce((a,b) => a || b, false)) {
        indicator.classList.add('labeled');
    } else {
        indicator.classList.remove('labeled');
    }

    if (label == 'O' && val) {
        doc.labels = {'O': true};
    }

    if (label != 'O' && val) {
        doc.labels['O'] = false;
    }

    document.querySelectorAll('.label-button').forEach((el) => {
        if (doc.labels[el.attributes['data-label'].value]) {
            el.classList.add('selected');
        } else {
            el.classList.remove('selected');
        }
    });
};

app.nextDoc = function() {
    if (app.curindex < app.documents.length - 1) {
        app.focusDoc(app.curindex + 1)
    }
};

app.prevDoc = function() {
    if (app.curindex > 0) {
        app.focusDoc(app.curindex - 1)
    }
};

app.start = function() {
    return getJSON('/api/labelbatch').then((resp) => {
        app.processLabelBatch(resp);
        document.querySelector('body').style.display = 'block';
    }, (err) => {
        alert(err.error);
        if (err.error == 'Model is training') {
            window.location.replace('/static/trainview.html');
        }
    });
};

app.upload = function() {
    if (app.documents.some(d => Object.values(d.labels).reduce((a,b) => a || b, false))) {
        if (!confirm('Some documents not labeled; proceed with upload?')) {
            return;
        }
    }

    const obj = {'labeled_docs': []};
    obj['labeled_docs'] = app.documents.filter(d => Object.values(d.labels).reduce((a,b) => a || b, false))
        .map(d => {
            return {'filename': d.filename, 'labels': Object.entries(d.labels).filter(x => x[1]).map(x => x[0]) }
        });
    let xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/uploadbatch',  true);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    xhr.send(JSON.stringify(obj));

    xhr.onloadend = function () {
        const status = xhr.status;
        if (status == 200) {
            alert('Upload complete and training started');
            xhr.open('GET', '/api/starttrain');
            xhr.send();
            xhr.onloadend = () => { window.location.replace('/static/trainview.html'); }
        } else {
            alert('Unexpected error: ' + xhr.response)
        }
    };
};

app.start();
document.querySelector('.control-button:first-child').addEventListener('click', () => {
    app.prevDoc()
});
document.querySelector('.control-button:last-child').addEventListener('click', () => {
    app.nextDoc()
});
document.querySelectorAll('.label-button').forEach(el => el.addEventListener('click', () => {
    if (el.classList.contains('selected')) {
        app.setLabel(el.attributes['data-label'].value, false);
    } else {
        app.setLabel(el.attributes['data-label'].value, true);
    }
}));
document.querySelector('#submit').addEventListener('click', () => app.upload());
