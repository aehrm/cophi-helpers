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
                    browserifyOptions: {
                        debug: true,
                        insertGlobalVars: {
                            GIT_VERSION: function(file, dir) {
                                revision = require('child_process')
                                    .execSync('git describe --dirty=-d --always')
                                    .toString().trim();
                                return '"' + revision + '"';
                            }
                        }
                    },
                    transform: [
                        ['node-underscorify', { "requires": [{"variable": "_", "module": "underscore"}]}]
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
        assets_inline: {
            dist: {
                files: {'dist/app.min.html': 'dist/app.html'}
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
    grunt.loadNpmTasks('grunt-assets-inline');

    grunt.registerTask('default', ['copy:dist', 'browserify:dist', 'stylus:dist', 'assets_inline:dist']);

};
