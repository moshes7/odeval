import os
import dill
import shutil

from analyze.analyzer import Analyzer
from analyze.tests.utils_test import compare_instances


def test_load_and_save():

    """
    Load, save and load again full and thin analyzer instances, and verify that content is identical.
    """
    base_dir = os.path.dirname(__file__)
    relative_data_dir = 'data/1_example'
    data_dir = os.path.join(base_dir, relative_data_dir)

    # ----------------------------
    # load and save full analyzer
    # ----------------------------

    # load analyzer
    analyzer_file = os.path.join(data_dir, 'analyzer_full.p')
    analyzer_full = Analyzer.load(analyzer_file)

    # save analyzer in temporary directory
    output_dir = os.path.join(data_dir, 'tmp_full')
    shutil.rmtree(output_dir, ignore_errors=True)  # delete existing temp dir

    # save instance with image dir
    output_file_full = os.path.join(output_dir, 'analyzer.p')
    analyzer_full.save(output_file_full, save_thin_instance=False, save_images_in_dir=True, image_name_template='{:08d}.png')

    # ----------------------------
    # load and save thin analyzer
    # ----------------------------
    analyzer_file = os.path.join(data_dir, 'analyzer_thin.p')
    analyzer_thin = Analyzer.load(analyzer_file, load_images_from_dir=True, sfx='png')  # use lossless png image, not jpg

    # save analyzer in temporary directory
    output_dir = os.path.join(data_dir, 'tmp_thin')
    shutil.rmtree(output_dir, ignore_errors=True)  # delete existing temp dir

    # save instance with image dir
    output_file_thin = os.path.join(output_dir, 'analyzer.p')
    analyzer_thin.save(output_file_thin, save_thin_instance=True, save_images_in_dir=True, image_name_template='{:08d}.png')  # use lossless png image, not jpg

    # ----------------------------------------------------------------
    # load analyzers from saved dirs and compare content thin analyzer
    # ----------------------------------------------------------------

    # load analyzers
    analyzer_full2 = Analyzer.load(output_file_full)
    analyzer_thin2 = Analyzer.load(output_file_thin, load_images_from_dir=True, sfx='png')  # use lossless png image, not jpg

    # compare content
    is_equal = compare_instances(analyzer_full2, analyzer_thin2)

    assert is_equal


if __name__ == '__main__':

    test_load_and_save()

    print('Done!')