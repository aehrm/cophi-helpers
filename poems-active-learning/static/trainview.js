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

const Eta = function (total, unit, start) {
    this.starttime = new Date();
    this.unit = unit;
    this.total = total;
    this.n = 0;

    if (start) this.start = start;
    else this.start = 0;
};
Eta.prototype.update = function (n) {
    this.n = n;
};
Eta.prototype.progress = function () {
    return this.n / this.total;
};
Eta.prototype.poststr = function () {
    const running = new Date() - this.starttime;
    const rate = (this.n - this.start) / running;
    const remaining = (this.total - this.n) / rate;

    if (Number.isNaN(remaining) || !Number.isFinite(remaining)) {
        return '';
    }

    function formatTime(t) {
        return `${Math.floor(t / 1000 / 60).toString()}:${Math.floor(t / 1000 % 60).toString().padStart(2, '0')}`;
    }

    return `(${formatTime(running)}<${formatTime(remaining)}, ${(rate * 1000).toFixed(2)} ${this.unit}/s)`
};


const getData = () => getJSON('/api/status').then((res) => {
    data = res;
    if (data.train_info)
        data.train_info.batch_size = data.train_info.train_data / data.train_info.batches;

    if (data.result)
        data.result.accuracy = {'f1-score': data.result.accuracy, 'support': data.result["macro avg"].support}

});

function createPlot() {
    const traces = [
        {x: [], y: [], name: 'Train Loss'},
        {x: [], y: [], name: 'Validation Loss'},
    ];
    const layout = {
        margin: {
            l: 50, t: 0, b: 20
        },
        height: 300,
        showlegend: true,
        legend: {
            x: 1,
            xanchor: 'left',
            y: 1
        },
        xaxis: {
            type: 'linear',
            autorange: true,
            tickmode: 'array',
            tickvals: [...Array(data.train_info.epochs + 1).keys()].map(x => x * data.train_info.batches),
            ticktext: [...Array(data.train_info.epochs).keys()].map(x => (x + 1).toString()).concat('')
        },
        yaxis: {
            type: 'log',
            autorange: true,
            tickformat: '.2e'
        }
    };

    Plotly.newPlot('trainchart', traces, layout, {staticPlot: true});
}

function updatePlot() {
    const curdata = document.getElementById('trainchart').data;
    const newTrainY = data.train_info.train_loss.slice(curdata[0].x.length);
    const newTrainX = Array.from([...Array(newTrainY.length)].keys()).map(x => x + curdata[0].x.length);

    const newValY = data.train_info.val_loss.slice(curdata[1].x.length);
    const newValX = Array.from([...Array(newValY.length)].keys()).map(x => data.train_info.batches * (1 + x + curdata[1].x.length));
    Plotly.extendTraces('trainchart', {x: [newTrainX, newValX], y: [newTrainY, newValY]}, [0, 1]);
}

function createEvalTable() {

    let table = document.querySelector('#evalTable');
    let tableBody = document.createElement('tbody');
    tableBody.innerHTML += '<thead><tr><th></th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr></thead>';

    [...Object.keys(data.result)].forEach((rowKey) => {
        let row = document.createElement('tr');
        row.innerHTML = `<td>${rowKey}</td>`;

        ['precision', 'recall', 'f1-score', 'support'].forEach(function (colKey) {
            if (data.result[rowKey][colKey] !== undefined) {
                const digits = colKey == 'support' ? 0 : 2;
                row.innerHTML += `<td>${data.result[rowKey][colKey].toFixed(digits)}</td>`
            } else {
                row.innerHTML += `<td></td>`;
            }
        });

        tableBody.appendChild(row);
    });

    table.appendChild(tableBody);
}

function updateTrainTable() {
    if (!trainEta) trainEta = new Eta(data.train_info.batch_size * data.train_info.batches * data.train_info.epochs, 'ex', data.train_info.batch_size * data.train_info.train_loss.length);
    const fields = document.querySelectorAll('#train-table tr td:last-child');

    const batchLosses = data.train_info.train_loss.slice(data.train_info.batch_size * Math.floor((data.train_info.train_loss.length - 1) / data.train_info.batch_size));
    fields[0].innerHTML = data.train_info.train_data;
    fields[1].innerHTML = data.train_info.val_data;
    if (batchLosses.length > 0)
        fields[2].innerHTML = (batchLosses.reduce((a, b) => a + b, 0) / batchLosses.length).toExponential(3);
    if (data.train_info.val_loss.length > 0)
        fields[3].innerHTML = data.train_info.val_loss.slice(-1)[0].toExponential(3);

    trainEta.update(data.train_info.batch_size * data.train_info.train_loss.length);
    const progress = trainEta.progress();
    const round = Math.min(data.train_info.val_loss.length + 1, data.train_info.epochs);

    document.querySelector('#train-progress .percentage').innerHTML = (100 * progress).toFixed(0) + '%';
    document.querySelector('#train-progress .bar').value = (100 * progress).toFixed(0);
    document.querySelector('#train-progress .post').innerHTML = `Epoch ${round}/${data.train_info.epochs} ${trainEta.poststr()}`;
}

function doWhilePromise(cond, delay, prom) {
    let whilst = () => {
        if (!cond()) return Promise.resolve();

        return new Promise((res) => {
            setTimeout(res, delay)
        }).then(prom).then(whilst);
    };

    return Promise.resolve().then(prom).then(whilst)
}


const updateTrain = () => doWhilePromise(() => data.stage == 'train', 1000, () => getData().then(() => {
    updatePlot();
    updateTrainTable();
}));

const updatePrediction = () => doWhilePromise(() => data.stage == 'predict', 1000, () => getData().then(() => {
    if (!predEta && data.predict_info.labeled > 0) predEta = new Eta(data.predict_info.documents, 'docs', data.predict_info.labeled);
    if (!predEta) return;

    predEta.update(data.predict_info.labeled);
    document.querySelector('#prediction-progress .percentage').innerHTML = (100 * predEta.progress()).toFixed(0) + '%';
    document.querySelector('#prediction-progress .bar').value = (100 * predEta.progress()).toFixed(0);
    document.querySelector('#prediction-progress .post').innerHTML = `${data.predict_info.labeled}/${data.predict_info.documents} ${predEta.poststr()}`;
}));

let data = null;
let trainEta = null;
let predEta = null;
getData().then(() => {
    document.querySelector('#start-date').innerHTML = new Date(data.startdate).toLocaleString();
})
    .then(() => doWhilePromise(() => data.stage == 'init', 1000, () => getData()))
    .then(() => {
        document.querySelector('#init-section .spinner').style.display = 'none';
        document.querySelector('#train-section').style.display = 'block';
        createPlot();
    })
    .then(updateTrain)
    .then(() => {
        document.querySelector('#train-section .spinner').style.display = 'none';
        document.querySelector('#predict-section').style.display = 'block';
    })
    .then(updatePrediction)
    .then(() => {
        document.querySelector('#predict-section .spinner').style.display = 'none';
        document.querySelector('#eval-section').style.display = 'block';
        document.querySelector('#end-date').innerHTML = new Date(data.enddate).toLocaleString();
        createEvalTable()
    });
