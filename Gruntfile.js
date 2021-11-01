module.exports = function(grunt) {

    grunt.initConfig({
        pkg: grunt.file.readJSON('package.json'),
        copy: {
            dist: {
                files: [ {expand: true, cwd: 'src/assets/', src: ['**'], dest: 'dist/'} ]
            }
        },
        browserify: {
            dist: {
                dest: 'dist/bundle.js',
                src: ['src/javascript/app.js'],
                options: {
                    compress: false,
                    browserifyOptions: { debug: true },
                    transform: [
                        ['node-underscorify', { "requires": [{"variable": "_", "module": "underscore"}, {"variable": "child_process", "module": "child_process"}]}]
                    ]
                }
            }
        },
        stylus: {
            dist: {
                options: {
                    compress: false,
                    'include css': true,
                    // use: [
                    //     function() { return require('autoprefixer-stylus')(); }
                    // ]
                },
                files: {
                    'dist/style.css': 'src/stylus/main.styl'
                }
            }
        },
        watch: {
            dist: {
                files: ['src/**', 'assets/**'],
                tasks: ['default']
            }
        }
    });

    grunt.loadNpmTasks('grunt-contrib-copy');
    grunt.loadNpmTasks('grunt-browserify');
    grunt.loadNpmTasks('grunt-contrib-stylus');
    grunt.loadNpmTasks('grunt-contrib-watch');

    grunt.registerTask('default', ['copy:dist', 'browserify:dist', 'stylus:dist']);

};
