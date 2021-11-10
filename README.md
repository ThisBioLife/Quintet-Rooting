# Rooting Species Trees using Phylogenetic Invariants

## Dependencies
- Python 3.x
- [Dendropy 4.x](https://dendropy.org/index.html)
- [Numpy](https://numpy.org)

To install the above dependencies, you can use `pip install -r requirements.txt`. 

## Usage
### Inferring rooted species tree
```
$ python3 rooting.py -i input_file -o output_file -m n
```
**Arguments**
```
 -h, --help       show this help message and exit
 -i, --input      input gene trees in newick format
 -o, --output     inferred rooted species tree in newick format
 -m, --mode       'n' stands for naive (scoring function approach) and 'c' stands for clustering
```
### Extracting subtree from larger tres by taxa
```
extract-quintet.py [-h] -i INPUT -o OUTPUT -if INPUT_FILENAME [-of OUTPUT_FILENAME] -t TAXA [TAXA ...]
```
**Arguments**
```
 -h, --help					        show this help message and exit
 -i, --input      			   input folder path (subfolders per model condition)
 -if, --input_filename		name of file containing input trees in each replicate
 -of, --output_filename	name of file containing output trees in each replicate
 -t, --taxa					        list of taxa in extracted subtrees
```
