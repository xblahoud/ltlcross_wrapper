# ltlcross wrapper
Python wrapper around the amazing tool `ltlcross` from [Spot](https://spot.lrde.epita.fr/) library.
The tool compares LTL to automata translators.

## Requires
* Python >= 3.6
* [Spot](https://spot.lrde.epita.fr/)
* [pandas](https://pandas.pydata.org/) >= 0.24
* [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org)
* [pandas2pgfplots](https://github.com/xblahoud/pandas2pgfplots)1

The following libraries are needed for bokeh scatter plots (can be used in Jupyter)
* [bokeh](https://bokeh.org/) (installs automatically by `pip`)
* [colorcet](https://colorcet.holoviz.org/) (installs automatically by `pip`)
* [jupyter_bokeh](https://github.com/bokeh/jupyter_bokeh) for rendering the plots in JupyterLab

## Installation
```
python3 -m pip install -U ltlcross_wrapper
```

## Usage
ltlcross_wrapper offers 3 classes:
 * `Modulizer` and `GoalModulizer` split a big `ltlcross` task into smaller ones, execute
    the small tasks in parallel, and merge the intermediate results into one final `.csv`,
    `.log`, and `_bogus.ltl` files. Always use `GoalModulizer` if one of the compared
    tools is [GOAL]!
 * `ResAnalyzer` parses the results of `ltlcross`, and implements several functions
    to analyze and visualize the results, mainly in Jupyter notebooks. 

### Modulizer
We need to specify the tools and file(s) with formulas which ltlcross should 
use. The tools are given as a dict whose items are pairs `(name, ltlcross_cmd)` 
where `ltlcross_cmd` is `COMMANDFMT` from `man ltlcross`.
 
    tools = {"LTL3HOA"    : "ltl3hoa -d -x -i -p 2 -f %f > %O",
             "SPOT"       : "ltl2tgba"
            }

Typical usage of `Modulizer` follows.

    m = Modulizer(tools, formulas.ltl)
    mp.run()
 
 The above command splits the file `formulas.ltl` into several files with
 2 formulas each, and uses 4 processes to run ltlcross on these small files
 in parallel. The number of processes can be controlled by setting 
 `processes` in both the constructor and the function `run()`. The number
 of formulas in each small file can be changed by setting `chunk_size`
 in the constructor.
 
 If a previous computation was interrupted for some reason, consider using
    
    m.resume()
 instead of `m.run()`. The function `m.resume()` will skip computing the
 small tasks for which it already finds an `.csv` file with the result.
 
 You can delete the final results and intermediate files, and rerun the
 computation by
 
    m.recompute()
    
By default, all intermediate files will be created in directory 
`modular.parts` and the merged files `modular.csv`, `modular.log`,
`modular_bogus.ltl` will be created in the current directory. The base
`modular` can be changed to `new_name` simply by setting `name=new_name`.
Otherwise, each filename/dirname can be changed by setting `out_res_file`, 
`out_log_file`, `out_bogus_file`, and `tmp_dir`.

#### Temporary files in ltlcross commands
Some tools need to read the input from (or store the output in) a
temporary file that is different from the one that ltlcross expects.
This file name has to be specified in the command for `ltlcross`. As
the task will be processed in parallel (unless requested otherwise
with `processes=1`), we can have a lot of race conditions. For this
purpose, each process created by the `Modulizer` class set its own
value into an environment variable called `$LCW_TMP`. You can then
specify tools with commands like

    "tool1" : ltl2tgba %f > $LCW_TMP.in.hoa && tool1 $LCW_TMP.in.hoa ...

### [GOAL] \& GoalModulizer
Always use `GoalModulizer` instead of `Modulizer` if you need to run
[GOAL] in parallel. `GoalModulizer` requires additional parameter
`goal_root` to specify the path to root directory of [GOAL] (contains
`gc`, `goal`, `boot.properties`). `GoalModulizer` uses a unique binary
and a unique shadow folder for each process that runs the task.
The path to the unique binary is stored in environment variable
`$LCW_GOAL_BIN` which you can use in specification of tools to compare.
The `goal_root` part of the path should be omitted in the ltlcross
command specification. Specifying [GOAL] is as simple as

    'my_goal_batch` : '$LCW_GOAL_BIN batch "{goals commands}"'

If you are using the batch mode of [GOAL], you often need to specify the
names of the temporary files in the [GOAL] command. The `$LCW_TMP` is
not recognized as a variable inside these commands. Use \`echo $LCW_TMP\`
which is a nested shell evaluation inside the [GOAL] command (enclosed in two \`).
See an example of complementation performed by GOAL and simplified by
Spot.

    tools = {
        "piterman": "ltl2tgba -B %f > $LCW_TMP.in && $LCW_GOAL_BIN batch '$temp = complement -m piterman `echo $LCW_TMP.in`; save -c HOAF $temp `echo $LCW_TMP.out;' && autfilt --small --tgba $LCW_TMP.out > %O
        "SPOT": "ltl2tgba -B %f | autfilt --complement > %O"
    }
    m = ltlcross_wrapper.GoalModulizer(goal_root="PATH_TO_GOAL", tools=tools, formula_file="MY_FORMS.ltl")
    m.run()

TODO: Explain ltlcross options

### Results' Analyzer

See the [usage notebook](Usage.ipynb). Currently, bokeh scatter plots do not
render directly on github so you might consider to [see the notebook on nbviewer](https://nbviewer.jupyter.org/github/xblahoud/ltlcross_wrapper/blob/master/Usage.ipynb)

[GOAL]: http://goal.im.ntu.edu.tw
