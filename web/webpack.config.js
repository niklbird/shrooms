const path = require('path');

module.exports = {
  entry: './main.js',
  output: {
    filename: 'main2.js',
    path: path.resolve(__dirname, 'dist'),
  },
};