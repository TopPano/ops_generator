const utils = require('./utils');
const lodash = require('lodash');
const changeCase = require('change-case');

/*
let shape = [ "vector:none", "vector:none", "CV_64FC3" ];

const lastVecIdx = utils.lastIndexOf(shape, (obj) => { return obj.startsWith('vector'); });


function declareVec(layer, core) {
  if (layer === 0) {
    return `${core}`;
  }
  return `vector<${declareVec(--layer, core)}>`;
}

console.log(declareVec(lastVecIdx + 1, 'Mat'));

const a1 = 'CV_64FC3';
const a2 = '64FC3';
const a3 = 'CV_128FC3';
const a4 = 'CV_64FC0';
const a5 = [ 'aaa' ];
const a6 = [ 'CV_64FC3', 'CV_8UC2' ];
const a7 = 'int';
const a8 = 'CV_16FC23';

console.log('a1: ' + JSON.stringify(utils.parseDataDtor(a1)));
console.log('a2: ' + JSON.stringify(utils.parseDataDtor(a2)));
console.log('a3: ' + JSON.stringify(utils.parseDataDtor(a3)));
console.log('a4: ' + JSON.stringify(utils.parseDataDtor(a4)));
console.log('a5: ' + JSON.stringify(utils.parseDataDtor(a5)));
console.log('a6: ' + JSON.stringify(utils.parseDataDtor(a6)));
console.log('a7: ' + JSON.stringify(utils.parseDataDtor(a7)));
console.log('a8: ' + JSON.stringify(utils.parseDataDtor(a8)));

const b1 = 'vector:none';
const b2 = 'vector:3';
const b3 = '3';
const b4 = '0';
const b5 = 'vector:0';
const b6 = 'vector';
const b7 = 'vector:';

console.log('b1: ' + JSON.stringify(utils.parseDimDtor(b1)));
console.log('b2: ' + JSON.stringify(utils.parseDimDtor(b2)));
console.log('b3: ' + JSON.stringify(utils.parseDimDtor(b3)));
console.log('b4: ' + JSON.stringify(utils.parseDimDtor(b4)));
console.log('b5: ' + JSON.stringify(utils.parseDimDtor(b5)));
console.log('b6: ' + JSON.stringify(utils.parseDimDtor(b6)));
console.log('b7: ' + JSON.stringify(utils.parseDimDtor(b7)));
*/

console.log(lodash.range(1));

//console.log(utils.expandArgusWithNumber(3, (idx) => { return `a(i, j, ${idx})`; }));
