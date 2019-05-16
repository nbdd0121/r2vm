let fs = require('fs');

let lines = fs.readFileSync('trace', 'utf-8').split('\n');
let pc = '';
let graph = '';
let json = {};

for (let line of lines) {
    if (graph) {
        graph += line;
        if (line == '}') {
            json[pc] = graph;
            graph = '';
        }
        continue;
    }

    if (line.startsWith('IR for ')) {
        pc = line.replace('IR for ', '');
        continue;
    }

    if (line.startsWith('digraph ')) {
        graph += line;
    }
}

fs.writeFileSync('trace.html', fs.readFileSync(__dirname + '/template.html', 'utf-8').replace('/*PLACEHOLDER*/', 'let data=' + JSON.stringify(json) + ';'), 'utf-8');
