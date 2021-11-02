const _ = require('underscore')
const Backbone = require('backbone')
const $ = require('jquery')

const RADIO_PROMPTS = [
    { label: 'Inhaltlich', key: 'inhaltlich'},
    { label: 'Emotional', key: 'emotional'},
    { label: 'Sprachlich', key: 'sprachlich'},
    { label: 'Formal', key: 'formal'},
    { label: 'Insgesamt', key: 'insgesamt'},
];

const DECISION_VALUES = [
    { label: 'Inhaltlich', key: 'inhaltlich'},
    { label: 'Emotional', key: 'emotional'},
    { label: 'Sprachlich', key: 'sprachlich'},
    { label: 'Formal', key: 'formal'}
];

const Prompt = Backbone.Model.extend({

    getTemplateObject: function(user) {
        if (!this.has('annotations')) {
            this.set('annotations', {});
        }
        if (!_.has(this.get('annotations'), user)) {
            this.get('annotations')[user] = {};
        }

        var o = _.clone(this.attributes);
        o.radioAnnotations = _.chain(RADIO_PROMPTS).map((p) => {
            p = _.clone(p);
            p.value = this.get('annotations')[user][p.key] || null
            return p;
        }).value();

        o.decisionAnnotation = {value: this.get('annotations')[user]['entscheidung'] || null};

        return o;
    },

    getAnnotation: function(user, key) {
        if (!this.has('annotations')) {
            return null;
        }
        if (!_.has(this.get('annotations'), user)) {
            return null;
        }

        return this.get('annotations')[user][key] || null;
    },

    setAnnotation: function(user, key, value) {
        if (!this.has('annotations')) {
            this.set('annotations', {});
        }
        if (!_.has(this.get('annotations'), user)) {
            this.get('annotations')[user] = {};
        }

        this.get('annotations')[user][key] = value;
        console.log(user, key, value);
        return this;
    },

    isDone: function(user) {
        const radioDone = _.chain(RADIO_PROMPTS).pluck('key').every((k) => {
            return this.getAnnotation(user, k) !== null
        }).value();

        return radioDone && this.getAnnotation(user, 'entscheidung') !== null;
    }

});

const AnnotationStore = Backbone.Model.extend({
    constructor: function(obj) {
        Backbone.Model.apply(this, arguments);
        this.set('prompts', new Backbone.Collection(obj.prompts || null, { model: Prompt }));
    }
});

const AppView = Backbone.View.extend({
    el: '.app-container',
    initialize: function() {
        this.showLogin();
    },
    setContent: function(view) {
        if (!!this.currentView)
            this.currentView.remove();
        this.currentView = view;
        this.currentView.render();
        this.$el.empty();
        this.$el.append(this.currentView.$el);
    },
    showLogin: function() {
        this.setContent(new StartView({app: this}));
    },
    showMain: function(args) {
        this.setContent(new AnnotationView(_.extend({app: this}, args)));
    }
});

const StartView = Backbone.View.extend({
    tagName: 'div',
    className: 'login',
    template: require('../templates/startview.html'),
    events: {
        'submit .start-form': 'form', 
        'click .start-header a[href]': 'toggleTheme'
    },
    render: function() {
        this.$el.html(this.template());
        return this;
    },
    initialize: function(options) {
        Object.assign(this, options);
    },
    form: function(ev) {
        ev.preventDefault();

        // Read File
        const file = this.$el.find('input[name="file"]')[0].files[0]
        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target.result;
            try {
                const parsed = JSON.parse(content);
                const store = new AnnotationStore(parsed);
                const username = this.$el.find('input[name="name"]').first().val();

                this.app.showMain({store: store, username: username, filename: file.name});
            } catch (e) {
                if (e instanceof SyntaxError) {
                    this.$el.find('.error').first().show();
                } else {
                    throw e;
                }
            }
        };
        reader.readAsText(file, 'UTF-8');

    },
    toggleTheme: function() {
        $('body').toggleClass('dark');
    }
});

const AnnotationView = Backbone.View.extend({
    tagName: 'div',
    className: 'annotation-view',
    template: require('../templates/annotation.html'),
    initialize: function(options) {
        Object.assign(this, options);

        this.modified = false;

        if (this.currentIndex === undefined) {
            this.gotoPrompt(_.find([...Array(this.store.get('prompts').length).keys()], (i) => !this.store.get('prompts').at(i).isDone(this.username)) || 0);
        } else {
            this.gotoPrompt(this.currentIndex);
        }

    },
    events: {
        //'focus .prompt-container input[type="radio"]':  'resetRadio',
        'click .prompt-container input[type="radio"]':  'processForm',
        'input .prompt-container input, .prompt-container select': 'processForm',
        'click .navigation button': 'processNavigation',
        'input .annotation-header select': 'processPromptSelection',
        'click .annotation-header button': 'processSave',
        'click .annotation-header a[href]': 'toggleTheme'
    },
    render: function() {
        this.$el.html(this.template({
            store: this.store,
            index: this.currentIndex,
            radioPrompts: RADIO_PROMPTS,
            decisionValues: DECISION_VALUES,
            username: this.username,
            prompt: this.store.get('prompts').at(this.currentIndex).getTemplateObject(this.username)
        }));
        return this;
    },
    processForm: function(ev) {
        console.log(ev.type)
        if (ev.type == 'click')
            this.lastClick = new Date().getTime();
        if (ev.type == 'input' && new Date().getTime() - this.lastClick < 100) {
            return;
        }

        const currentPrompt = this.store.get('prompts').at(this.currentIndex)
        const wasDone = currentPrompt.isDone(this.username);

        if ($(ev.target).is('input[type="radio"]')) {
            if (currentPrompt.getAnnotation(this.username, ev.target.name) == ev.target.value) {
                currentPrompt.setAnnotation(this.username, ev.target.name, null);
                ev.target.checked = false;
            } else {
                currentPrompt.setAnnotation(this.username, ev.target.name, ev.target.value);
            }
        } else if ($(ev.target).is('select') && ev.target.value !== 'same') {
            currentPrompt.setAnnotation(this.username, ev.target.name, ev.target.value === 'empty' ? null : ev.target.value);
            this.$el.find('input[type="text"]').hide();
        } else if ($(ev.target).is('select') && ev.target.value === 'same') {
            currentPrompt.setAnnotation(this.username, ev.target.name, null);
            this.$el.find('input[type="text"]').val('').show();
        } else if ($(ev.target).is('input[type="text"]')) {
            currentPrompt.setAnnotation(this.username, 'entscheidung', $(ev.target).val() || null);
        }

        if (wasDone !== currentPrompt.isDone(this.username)) {
            this.render();
        }

        this.modified = true;

    },
    gotoPrompt: function(i) {
        this.currentIndex = i;
        this.render();
    },
    processNavigation: function(ev) {
        var next;
        if (ev.target.name === 'next') {
            next = Math.min(this.store.get('prompts').length - 1, this.currentIndex + 1);
        } else {
            next = Math.max(0, this.currentIndex - 1);
        }

        this.gotoPrompt(next);
    },
    processPromptSelection: function(ev) {
        this.gotoPrompt(parseInt(ev.target.value))
    },
    processSave: function(ev) {
        function downloadBlob(blob, name = 'file.txt') {
            const blobUrl = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = blobUrl;
            link.download = name;
            document.body.appendChild(link);
            link.dispatchEvent(
                new MouseEvent('click', { 
                    bubbles: true, 
                    cancelable: true, 
                    view: window 
                })
            );
            document.body.removeChild(link);
        }

        if (ev.target.name == 'save') {
            const output = JSON.stringify(this.store)
            downloadBlob(new Blob([output]), this.store.get('description').title.replace(' ', '_') + '_' + this.username + '_' + new Date().toISOString().replace(/:[0-9]{2}\..*/, '') + '.json')
            this.modified = false;
        } else if (ev.target.name == 'end') {
            if (this.modified) {
                const conf = window.confirm('Änderungen wurden noch nicht gespeichert! Sicher, dass Du die Session beenden willst, und die Änderungen verlieren willst?')
                if (!conf) return;
            }

            this.app.showLogin();
        }

    },
    toggleTheme: function() {
        $('body').toggleClass('dark');
    }
});


$(() => {
    const appView = new AppView();
    window.addEventListener('beforeunload', (event) => {
        if (appView.currentView instanceof AnnotationView && appView.currentView.modified) {
            event.returnValue = 'You have unfinished changes!';
        }
    });
})
