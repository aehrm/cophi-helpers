var application = {
    currentDocument: null,
    segmentTreeLeft: null,
    segmentTreeRight: null,
    sentenceTree: null,
    colorMap: null,
};


application.openDocument = function(doc) {
    this.currentDocument = doc;

    this.segmentTreeLeft = new IntervalTree(Math.floor(this.currentDocument.text.length/2));
    this.segmentTreeRight = new IntervalTree(Math.floor(this.currentDocument.text.length/2));
    this.sentenceTree = new IntervalTree(Math.floor(this.currentDocument.text.length/2));

    Object.values(this.currentDocument.left_segments).forEach(seg => this.segmentTreeLeft.add(seg.start, seg.end-1, seg.id));
    Object.values(this.currentDocument.right_segments).forEach(seg => this.segmentTreeRight.add(seg.start, seg.end-1, seg.id));
    Object.values(this.currentDocument.sentences).forEach(s => this.sentenceTree.add(s.start, s.end-1));

    document.querySelector('#scoreselect').innerHTML = this.currentDocument.alignment_scores.map((x, i) => `<option value="${i}">${x.name}</option>`).join('');


    this.setText();
    this.colorSegments();
    this.setStripes();
    this.setScore(this.currentDocument.alignment_scores[0].name);
    this.adjustViewfinder(window.scrollY);
};

application.setText = function() {
    const rawText = this.currentDocument.text;
    const sents = this.currentDocument.sentences;
    const highlightText = (start, end, segments) => {
        if (segments.length == 0) {
            return rawText.substring(start, end);
        }

        var markup = rawText.substring(start, Math.max(segments[0].start, start));
        for (var i = 0; i < segments.length; i++) {
            markup += `<span class="segment-highlight" segment-id="${segments[i].id}">`;
            markup += rawText.substring(Math.max(segments[i].start, start), Math.min(segments[i].end, end));
            markup += `</span>`;

            markup += rawText.substring(Math.min(segments[i].end, end), i+1 < segments.length ? Math.min(segments[i+1].start, end) : end);
        };

        return markup;
    };

    const makeRow = (start, end, sentnum, sentidx) => {

        const leftSegments = this.segmentTreeLeft.rangeSearch(start, end-1).map(x => this.currentDocument.left_segments[x.id]);
        const rightSegments = this.segmentTreeRight.rangeSearch(start, end-1).map(x => this.currentDocument.right_segments[x.id]);
        const leftMarkup = highlightText(start, end, leftSegments);
        const rightMarkup = highlightText(start, end, rightSegments);


        return `<div class="sentence" index="${sentidx}">
            <div class="left">${leftMarkup}</div>
            <div class="score"></div>
            <div class="sentnum">${sentnum}</div>
            <div class="right">${rightMarkup}</div>
        </div>`
    };

    const textBuffer = []
    if (sents[0].start != 0) {
        textBuffer.push(makeRow(0, sents[0].start, 0, undefined));
    }
    for (var i = 0; i < sents.length; i++) {
        textBuffer.push(makeRow(sents[i].start, i+1 < sents.length ? sents[i+1].start : rawText.length, i+1, i));
    }

    document.querySelector('.text-container').innerHTML = textBuffer.join('');
};

application.colorSegments = function() {

    const colorSegments = (segments) => {
        segments.sort((a,b) => a.start - b .start);
        
        for (var i = 0; i < segments.length; i++) {
            this.colorMap[segments[i].id] = i%2;
            document.querySelectorAll(`.segment-highlight[segment-id="${segments[i].id}"]`).forEach(s => s.classList.add(`segment-highlight-${i%2}`));
        }
    };

    this.colorMap = {};
    //colorSegments(Object.values(this.currentDocument.left_segments));
    //colorSegments(Object.values(this.currentDocument.right_segments));

    const intersections = [];
    Object.values(this.currentDocument.left_segments).forEach(seg => {
        this.segmentTreeRight.rangeSearch(seg.start, seg.end-1).forEach(right => {
            const overlap = Math.min(seg.end-1, right.end-1) - Math.max(seg.start, right.start);
            intersections.push([seg.id, right.id, overlap]);
        });
    });

    const pathWeightedAnticlique = (weights) => {
        if (weights.length == 1) return [0];

        const A = [];
        A[0] = weights[0];

        for (var i = 1; i < weights.length; i++) {
            if (i > 1) {
                A[i] = Math.max( A[i-1], A[i-2]+weights[i] );
            } else {
                A[i] = Math.max( A[i-1], weights[i] );
            }
        }

        const anticlique = [];
        var i = A.length-1;
        while (i >= 0) {
            if (A[i] == A[i-1]) {
                i--;
            } else {
                anticlique.push(i);
                i -= 2;
            }
        }

        return anticlique;
    };

    const adjacentColor = (id, side) => {
        var pred;
        var succ;
        if (this.currentDocument.left_segments[id] != undefined) {
            const sentIdxStart = this.sentenceTree.pointSearch(this.currentDocument.left_segments[id].start)[0].id;
            const sentIdxEnd = this.sentenceTree.pointSearch(this.currentDocument.left_segments[id].end-2)[0].id;
            if (sentIdxStart - 1 >= 0) {
                const prevSent = this.currentDocument.sentences[sentIdxStart-1];
                pred = this.segmentTreeLeft.rangeSearch(prevSent.start, prevSent.end-1).reverse();
            }
            if (sentIdxEnd + 1 < this.currentDocument.sentences.length) {
                const nextSent = this.currentDocument.sentences[sentIdxEnd+1];
                succ = this.segmentTreeLeft.rangeSearch(nextSent.start, nextSent.end-1);
            }

        } else if (this.currentDocument.right_segments[id] != undefined) {
            const sentIdxStart = this.sentenceTree.pointSearch(this.currentDocument.right_segments[id].start)[0].id;
            const sentIdxEnd = this.sentenceTree.pointSearch(this.currentDocument.right_segments[id].end-2)[0].id;
            if (sentIdxStart - 1 >= 0) {
                const prevSent = this.currentDocument.sentences[sentIdxStart-1];
                pred = this.segmentTreeRight.rangeSearch(prevSent.start, prevSent.end-1).reverse();
            }
            if (sentIdxEnd + 1 < this.currentDocument.sentences.length) {
                const nextSent = this.currentDocument.sentences[sentIdxEnd+1];
                succ = this.segmentTreeRight.rangeSearch(nextSent.start, nextSent.end-1);
            }
        } else {
            throw new Error();
        }

        var predColor = undefined;
        if (pred && pred.length > 0) {
            predColor = this.colorMap[pred[0].id];
        }
        var succColor = undefined;
        if (succ && succ.length > 0) {
            succColor = this.colorMap[succ[0].id];
        }

        return [predColor, succColor].filter(c => c != undefined);
    };

    var path = [intersections[0]];
    var i = 1;
    while (i < intersections.length) {
        path = [intersections[i]];
        i++;

        while (i < intersections.length) {
            var pred = path[path.length-1];
            var cur = intersections[i];
            if (pred[0] == cur[0] || pred[1] == cur[1]) {
                path.push(cur);
                i++;
            } else {
                break;
            }
        }

        var anticlique = pathWeightedAnticlique(path.map(x=>x[2]));

        var components = [];
        anticlique.reverse().forEach(i => {
            var component = components.filter(x => x.has(path[i][0]) || x.has(path[i][1]))[0];

            if (component == undefined) {
                components.push(new Set([path[i][0], path[i][1]]));
            } else {
                component.add(path[i][0]);
                component.add(path[i][1]);
            }
        });

        components.forEach(component => {
            var colors = [0,1,2];
            component.forEach(id => {
                colors = colors.filter(c => !adjacentColor(id).includes(c));
                colors = colors.filter(c => !adjacentColor(id).includes(c));
            });

            component.forEach(i => {
                this.colorMap[i] = colors[0];
                this.colorMap[i] = colors[0];
            });
        });


        path = [];
    }

    [Object.keys(this.currentDocument.left_segments), Object.keys(this.currentDocument.right_segments)].flat().forEach(id => {
        if (this.colorMap[id] == undefined) {
            this.colorMap[id] = [0,1,2].filter(c => !adjacentColor(id, 'right').includes(c))[0];
        }
    });

    document.querySelectorAll(`.segment-highlight`).forEach(el => {
        const id = el.getAttribute('segment-id');
        el.classList.add(`segment-highlight-${this.colorMap[id]}`)
    });

};

application.setStripes = function() {
    const textlen = this.currentDocument.text.length;

    const setSegmentStripes = (container, segments) => {
        const prewidth = segments.length > 0 ? segments[0].start/textlen : 1;
        const textBuffer = [];
        textBuffer.push(`<div class="stripe" style="width: ${prewidth*100}%"></div>`);
        for (var i = 0; i < segments.length; i++) {
            const hi = this.colorMap[segments[i].id];
            const width = (segments[i].end - segments[i].start)/textlen;
            const postwidth = i+1<segments.length ? (segments[i+1].start - segments[i].end)/textlen : 0;

            textBuffer.push(`<div class="stripe stripe-highlight-${hi}" style="width: ${width*100}%"></div>`);
            textBuffer.push(`<div class="stripe" style="width: ${postwidth*100}%"></div>`);
        }

        container.innerHTML = textBuffer.join('');
    };

    setSegmentStripes(document.querySelector('.left-stripes'), Object.values(this.currentDocument.left_segments));
    setSegmentStripes(document.querySelector('.right-stripes'), Object.values(this.currentDocument.right_segments));

    const textBuffer = [];
    const barWidth = document.querySelector('.sentence-stripes').offsetWidth;
    const sentences = this.currentDocument.sentences;
    const prewidth = sentences[0].start/textlen;
    textBuffer.push(`<div class="stripe" style="width: ${prewidth*100}%"></div>`);
    for (var i = 0; i < sentences.length; i++) {
        if (i + 1 < sentences.length) {
            var width = (sentences[i+1].start - sentences[i].start)/textlen;
        } else {
            var width = (textlen - sentences[i].start)/textlen;
        }

        textBuffer.push(`<div class="stripe" index="${i}" style="width: ${width*barWidth}px"></div>`);
        //textBuffer.push(`<div class="stripe" index="${i}" style="width: ${width*100}%"></div>`);
    }
    document.querySelector('.sentence-stripes').innerHTML = textBuffer.join('');

};

application.setScore = function(name) {
    const perc2color = (perc) => {
            var r, g, b = 0;
            if(perc < 50) {
                        r = 255;
                        g = Math.round(5.1 * perc);
                    }
            else {
                        g = 255;
                        r = Math.round(510 - 5.10 * perc);
                    }
            const h = r * 0x10000 + g * 0x100 + b * 0x1;
            return '#' + ('000000' + h.toString(16)).slice(-6);
    }

    const scores = this.currentDocument.alignment_scores.find(x => x.name == name).scores;

    document.querySelectorAll(`.sentence > .score`).forEach(el => {
        const i = parseInt(el.parentNode.getAttribute('index'));
        el.innerHTML = (Math.round(scores[i]*100)/100).toFixed(2).toString();
        el.style['background-color'] = perc2color(scores[i]*100);
    });

    document.querySelectorAll(`.sentence-stripes > .stripe`).forEach(el => {
        const i = parseInt(el.getAttribute('index'));
        el.style['background-color'] = perc2color(scores[i]*100);
    });
};

application.move = function(fraction) {
    //var textContainer = document.querySelector('.text-container');
    //window.scrollTo(0, fraction * textContainer.offsetHeight + textContainer.offsetTop - document.documentElement.clientHeight/2);
    const charIndex = fraction * this.currentDocument.text.length;
    const pos = Math.max(0, this.sentenceTree.pointTree.firstPositionOf(new Point(Math.round(charIndex))));

    const sentenceIndex = this.sentenceTree.pointTree[pos].id;
    const sentenceEl = document.querySelector(`.sentence[index="${sentenceIndex}"]`);

    window.scrollTo(0, sentenceEl.offsetTop - document.documentElement.clientHeight/2);
};

application.adjustViewfinder = function(scrollPos) {
    const viewfinder = document.querySelector('.viewfinder');
    const barWidth = document.querySelector('.sentence-stripes').offsetWidth
    const textContainer = document.querySelector('.text-container');
    //viewfinder.style.width = ((document.documentElement.clientHeight-100)/textContainer.offsetHeight * barWidth) + 'px';
    //viewfinder.style.display = 'block';
    //viewfinder.style.left = pos + 'px';
    

    const textLen = this.currentDocument.text.length;
    var leftBorder = 0;
    var rightBorder = 1;
    

    var topElement = document.elementFromPoint(100, 110);
    if (topElement.classList.contains('segment-highlight')) {
        topElement = topElement.parentNode.parentNode;
    } else if (topElement.classList.contains('left')) {
        topElement = topElement.parentNode;
    } else {
        topElement = null;
    }

    var bottomElement = document.elementFromPoint(100, document.documentElement.clientHeight - 10);
    if (bottomElement.classList.contains('segment-highlight')) {
        bottomElement = bottomElement.parentNode.parentNode;
    } else if (bottomElement.classList.contains('left')) {
        bottomElement = bottomElement.parentNode;
    } else {
        bottomElement = null;
    }

    if (topElement) {
        leftBorder = this.currentDocument.sentences[topElement.getAttribute('index')].start / textLen;
    }
    if (bottomElement) {
        rightBorder = this.currentDocument.sentences[bottomElement.getAttribute('index')].end / textLen;
    }

    viewfinder.style.left = (leftBorder * barWidth) + 'px';
    viewfinder.style.right = (barWidth - rightBorder * barWidth) + 'px';
    viewfinder.style.display = 'block';
};



document.querySelectorAll('.stripes').forEach(el => el.addEventListener('mousedown', ev => {
    application.move((ev.clientX - el.getBoundingClientRect().left)/el.offsetWidth);
}));

document.querySelector('#scoreselect').addEventListener('change', ev => {
    const i = parseInt(ev.target.value);
    application.setScore(application.currentDocument.alignment_scores[i].name);
});

document.querySelector('#fileinput').addEventListener('change', ev => {
    const reader = new FileReader();
    reader.onload = function(e) {
        const doc = JSON.parse(reader.result);
        application.openDocument(doc);
    }

    if (ev.target.files.length > 0) {
        reader.readAsText(ev.target.files[0]);
    }
});

document.addEventListener('scroll', ev => {
    application.adjustViewfinder(window.scrollY);
});
