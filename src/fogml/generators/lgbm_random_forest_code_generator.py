import lightgbm as lgbm
from .base_generator import BaseGenerator


class LGBMRandomForestCodeGenerator(BaseGenerator):
    def __init__(self, clf):
        if isinstance(clf, lgbm.LGBMClassifier):
            self.clf = clf.booster_
        else:
            self.clf = clf

    def generate_statements(self, tree, index):

        def recurse(node, depth):
            indent = "  " * depth
            if 'leaf_index' not in node:
                name = node['split_feature']
                threshold = node['threshold']
                return indent + "if (x[%d] <= " % name + "%.10f" % threshold + ") {\n" + \
                       recurse(node['left_child'], depth + 1) + \
                       indent + "}\n" + indent + "else {\n" + \
                       recurse(node['right_child'], depth + 1) + \
                       indent + "}\n"
            else:
                return indent + 'results[%s] = %s;\n' % (index, str(node['leaf_value']))

        return recurse(tree, 1)

    def generate(self, fname='lgbm_random_forest_model.c', cname="classifier", **kwargs):
        model_structure = self.clf.dump_model()
        estimators = model_structure['tree_info']
        result = self.license_header()
        result += "int %s(double * x){\n" % cname
        result += "  int results[%s];\n" % len(estimators)
        index = 0
        for estimator in estimators:
            result += self.generate_statements(estimator['tree_structure'], index)
            index += 1
        if model_structure['num_class'] > 1:
            result += "  double proba[%s];\n" % model_structure['num_class']
            result += "  for (int i = 0; i < %s; i++){\n" % model_structure['num_class']
            result += "    proba[i] = 0;\n"
            result += "  }\n"
            result += "  for (int i = 0; i < %s; i++){\n" % len(estimators)
            result += "    proba[i %% %s] += results[i];\n" % model_structure['num_class']
            result += "  }\n"
            result += "  for (int i = 0; i < %s; i++){\n" % model_structure['num_class']
            result += "    proba[i] = exp(proba[i]);\n"
            result += "  }\n"
            result += "  double sum = 0;\n"
            result += "  for (int i = 0; i < %s; i++){\n" % model_structure['num_class']
            result += "    sum += proba[i];\n"
            result += "  }\n"
            result += "  for (int i = 0; i < %s; i++){\n" % model_structure['num_class']
            result += "    proba[i] /= sum;\n"
            result += "  }\n"
            result += "  int index = 0;\n"
            result += "  double max = proba[0];\n"
            result += "  for (int i = 0; i < %s; i++){\n" % model_structure['num_class']
            result += "    if (proba[i] > max){\n"
            result += "      max = proba[i];\n"
            result += "      index = i;\n"
            result += "    }\n"
            result += "  }\n"
            result += "  return index;\n"
        else:
            result += "  double proba = 0;\n"
            result += "  for (int i = 0; i < %s; i++){\n" % len(estimators)
            result += "    proba += results[i];\n"
            result += "  }\n"
            result += "  proba = 1 / (1 + exp(-1 * proba));\n"
            result += "  if (proba > 0.5){\n"
            result += "    return 1;\n"
            result += "  }\n"
            result += "  return 0;\n"
        result += "}\n"
        with open(fname, 'w') as c_file:
            c_file.write(result)
