# Requirements

- TensorFlow
- PyYAML
- Optional (for logging): slackclient

# Installation

1. Install the requirements.
2. Copy `config/private_example.yaml` to `config/private.yaml` and fill the missing values.
3. Prepare the data and embedding matrices in the folders specified in `private.yaml`
4. Use `train.sh` to start training. `config/config.py` has information about how to control various hyperparameters

# Data

You can contact me `matus.pikuliak@gmail.com` to ask for the data I used during my experiments.

# Papers

```
@inproceedings{DBLP:conf/slsp/PikuliakS20,
  author    = {Mat{\'{u}}s Pikuliak and
               Mari{\'{a}}n Simko},
  editor    = {Luis Espinosa Anke and
               Carlos Mart{\'{\i}}n{-}Vide and
               Irena Spasic},
  title     = {Exploring Parameter Sharing Techniques for Cross-Lingual and Cross-Task
               Supervision},
  booktitle = {Statistical Language and Speech Processing - 8th International Conference,
               {SLSP} 2020, Cardiff, UK, October 14-16, 2020, Proceedings},
  series    = {Lecture Notes in Computer Science},
  volume    = {12379},
  pages     = {97--108},
  publisher = {Springer},
  year      = {2020},
  url       = {https://doi.org/10.1007/978-3-030-59430-5\_8},
  doi       = {10.1007/978-3-030-59430-5\_8}
}



@inproceedings{DBLP:conf/tsd/PikuliakS20,
  author    = {Mat{\'{u}}s Pikuliak and
               Mari{\'{a}}n Simko},
  editor    = {Petr Sojka and
               Ivan Kopecek and
               Karel Pala and
               Ales Hor{\'{a}}k},
  title     = {Combining Cross-lingual and Cross-task Supervision for Zero-Shot Learning},
  booktitle = {Text, Speech, and Dialogue - 23rd International Conference, {TSD}
               2020, Brno, Czech Republic, September 8-11, 2020, Proceedings},
  series    = {Lecture Notes in Computer Science},
  volume    = {12284},
  pages     = {162--170},
  publisher = {Springer},
  year      = {2020},
  url       = {https://doi.org/10.1007/978-3-030-58323-1\_17},
  doi       = {10.1007/978-3-030-58323-1\_17}
}



@inproceedings{DBLP:conf/sofsem/PikuliakSB19,
  author    = {Mat{\'{u}}s Pikuliak and
               Mari{\'{a}}n Simko and
               M{\'{a}}ria Bielikov{\'{a}}},
  editor    = {Barbara Catania and
               Rastislav Kr{\'{a}}lovic and
               Jerzy R. Nawrocki and
               Giovanni Pighizzini},
  title     = {Towards Combining Multitask and Multilingual Learning},
  booktitle = {{SOFSEM} 2019: Theory and Practice of Computer Science - 45th International
               Conference on Current Trends in Theory and Practice of Computer Science,
               Nov{\'{y}} Smokovec, Slovakia, January 27-30, 2019, Proceedings},
  series    = {Lecture Notes in Computer Science},
  volume    = {11376},
  pages     = {435--446},
  publisher = {Springer},
  year      = {2019},
  url       = {https://doi.org/10.1007/978-3-030-10801-4\_34},
  doi       = {10.1007/978-3-030-10801-4\_34}
}


```
