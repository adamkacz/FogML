"""
   Copyright 2021 FogML

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import pandas

from xgboost import XGBRFClassifier, XGBRFRegressor, XGBClassifier
from .base_generator import BaseGenerator

class XGBoostRandomForestGenerator(BaseGenerator):
    def __init__(self, clf, tab='    '):
        self.clf = clf
        self.tab = tab
    

    def generate(self, fname = 'xgboost_random_forest_model.c', cname="classifier"):
        df = self.clf.get_booster().trees_to_dataframe()
        trees = df['Tree'].unique()
        
        feature_names = self.clf.get_booster().feature_names or [f"f{i}" for i in range(self.clf.n_features_in_)]
        self.feature_name_to_idx = { feature: idx for idx, feature in enumerate(feature_names)}
        
        code = ""
        code = self.license_header()
        for tree_idx in trees:
            nodes = df[df["Tree"] == tree_idx].set_index('ID')
            code += f"\n{self.traverse(tree_idx, nodes)}"
        #Must be int to initialise arays in C    
        number_classes = int(len(trees)/clf.get_num_boosting_rounds())
        
        if clf.objective == 'binary:logistic':
            code += f"""
    double leaves_sum = 0;
    double probability = 0;
    int result_class = -1;
    for(int i=0; i<{len(trees)}; i++){{
        leaves_sum += results[i];
    }}
    probability = 1/(1+ exp(-1 * leaves_sum));
    if(probability > 0.5){{
        result_class = 1;
    }}
    else{{
        result_class = 0;
    }}
    return result_class;
            
            
            """
        elif clf.objective == 'multi:softprob':
            code += f"""
    int classes_amount = {number_classes};
    double classes_values[{number_classes}]= {{0}};
    double sum = 0;
    for(int i=0; i<{len(trees)}; i++){{
        int current_class = 0;
        double current_value = 0;
        double new_value = 0;
        current_class = i%classes_amount ;
        current_value = classes_values[current_class];
        new_value = current_value + results[i];
        classes_values[current_class] = new_value;
        new_value = sum + results[i];
        sum = new_value;
    }};
    double function_values[{number_classes}]={{0}};
    double function_sum = 0;
    for(int j=0; j<{number_classes}; j++){{
        function_values[j] = exp(classes_values[j]);
        double new_sum = function_sum + function_values[j];
        function_sum = new_sum;
    }}
    double probabilities[{number_classes}] = {{0}};
    for(int k=0; k<{number_classes}; k++){{
        probabilities[k] = function_values[k]/function_sum;
    }}
    double max_probability = probabilities[0];
    int max_index = 0;
    for (int m = 0; m < {number_classes}; m++)
    {{
        if (probabilities[m] > max_probability) {{
            max_probability = probabilities[m];
            max_index = m;
        }}
    }}
    return max_index;
            
            """
      

        result = f"""
/*
Order of features: {feature_names}
*/
int {cname}(double * x) {{
    double results[{len(trees)}];
    {code}
    
}}
"""
        print(result)
        with open(fname, 'w') as c_file:
            c_file.write(result)

    def indent(self, string, depth=1):
        return ''.join([f"{self.tab * depth}{line}\n" for line in string.split('\n')])

    def traverse(self, tree_idx, nodes, depth=0):
        def rec(root_id, depth=depth):
            if root_id:
                feature, = nodes.loc[root_id, ['Feature']]

                if feature == 'Leaf':
                    gain, = nodes.loc[root_id, ['Gain']]
                    return f"results[{tree_idx}] = {gain};"
                else:
                    split, yes, no, missing = nodes.loc[root_id, ['Split', 'Yes', 'No', 'Missing']]
                    return self.indent(
f"""
if (x[{self.feature_name_to_idx[feature]}] && x[{self.feature_name_to_idx[feature]}] < {split}) {{
    {rec(yes, depth + 1)}
}} else {{
     {rec(no, depth + 1)}
}}""")
        

        return rec(nodes.index[0])