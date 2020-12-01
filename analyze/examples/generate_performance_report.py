import os

from analyze.analyzer import Analyzer

def generate_report():

    # load reference analyzer
    base_dir = os.path.dirname(__file__)
    relative_data_dir = '../tests/data/ILSVRC2015_00078000'
    data_dir = os.path.join(base_dir, relative_data_dir)
    analyzer_file = os.path.join(data_dir, 'analyzer.p')
    analyzer = Analyzer.load(analyzer_file, load_images_from_dir=False)

    # set output directory
    analyzer.output_dir = os.path.join(base_dir, 'output', 'generate_report')
    os.makedirs(analyzer.output_dir, exist_ok=True)

    # generate report
    analyzer.evaluate_performance(generate_report=True)

    pass

if __name__ == '__main__':

    generate_report()

    print('Done!')