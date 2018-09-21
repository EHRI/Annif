"""Definitions for command-line (Click) commands for invoking Annif
operations and printing the results to console."""


import collections
import os.path
import re
import sys
import click
import click_log
from flask.cli import FlaskGroup
import annif
import annif.corpus
import annif.eval
import annif.project
from annif.hit import HitFilter
from annif import logger

click_log.basic_config(logger)

cli = FlaskGroup(create_app=annif.create_app)


def get_project(project_id):
    """
    Helper function to get a project by ID and bail out if it doesn't exist"""
    try:
        return annif.project.get_project(project_id)
    except ValueError:
        click.echo(
            "No projects found with id \'{0}\'.".format(project_id),
            err=True)
        sys.exit(1)


def parse_backend_params(backend_param):
    """Parse a list of backend parameters given with the --backend-param
    option into a nested dict structure"""
    backend_params = collections.defaultdict(dict)
    for beparam in backend_param:
        backend, param = beparam.split('.', 1)
        key, val = param.split('=', 1)
        backend_params[backend][key] = val
    return backend_params


def generate_filter_batches():
    filter_batches = collections.OrderedDict()
    for limit in range(1, 16):
        for threshold in [i * 0.05 for i in range(20)]:
            hit_filter = HitFilter(limit, threshold)
            batch = annif.eval.EvaluationBatch()
            filter_batches[(limit, threshold)] = (hit_filter, batch)
    return filter_batches


@cli.command('list-projects')
def run_list_projects():
    """
    List available projects.

    Usage: annif list-projects
    """

    template = "{0: <15}{1: <30}{2: <15}"

    header = template.format("Project ID", "Project Name", "Language")
    click.echo(header)
    click.echo("-" * len(header))

    for proj in annif.project.get_projects().values():
        click.echo(template.format(proj.project_id, proj.name, proj.language))


@cli.command('show-project')
@click.argument('project_id')
def run_show_project(project_id):
    """
    Show project information.

    Usage: annif show-project <project_id>

    Outputs a human-readable string representation formatted as follows:

    Project ID:    testproj
    Language:      fi
    """

    proj = get_project(project_id)

    template = "{0:<20}{1}"

    click.echo(template.format('Project ID:', proj.project_id))
    click.echo(template.format('Project Name:', proj.project_id))
    click.echo(template.format('Language:', proj.language))


@cli.command('loadvoc')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('subjectfile', type=click.Path(dir_okay=False))
def run_loadvoc(project_id, subjectfile):
    proj = get_project(project_id)
    if annif.corpus.SubjectFileSKOS.is_rdf_file(subjectfile):
        # SKOS/RDF file supported by rdflib
        subjects = annif.corpus.SubjectFileSKOS(subjectfile, proj.language)
    else:
        # probably a TSV file
        subjects = annif.corpus.SubjectFileTSV(subjectfile)
    proj.load_vocabulary(subjects)


@cli.command('loaddocs')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('docfile', type=click.Path(dir_okay=False), nargs=-1)
def run_loaddocs(project_id, docfile):
    proj = get_project(project_id)
    if len(docfile) > 1:
        corpora = [annif.corpus.DocumentFile(docfn) for docfn in docfile]
        documents = annif.corpus.CombinedCorpus(corpora)
    else:
        documents = annif.corpus.DocumentFile(docfile[0])
    proj.load_documents(documents)


@cli.command('load')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('directory', type=click.Path(file_okay=False))
def run_load(project_id, directory):
    proj = get_project(project_id)
    subjects = annif.corpus.SubjectDirectory(directory)
    proj.load_subjects(subjects)


@cli.command('analyze')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
@click.option('--backend-param', '-b', multiple=True)
def run_analyze(project_id, limit, threshold, backend_param):
    """"
    Analyze a document.

    USAGE: annif analyze <project_id> [--limit=N] [--threshold=N] <document.txt
    """
    project = get_project(project_id)
    text = sys.stdin.read()
    backend_params = parse_backend_params(backend_param)
    hit_filter = HitFilter(limit, threshold)
    hits = hit_filter(project.analyze(text, backend_params))
    for hit in hits:
        click.echo("<{}>\t{}\t{}".format(hit.uri, hit.label, hit.score))


@cli.command('analyzedir')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('directory', type=click.Path(file_okay=False))
@click.option('--suffix', default='.annif')
@click.option('--force/--no-force', default=False)
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
@click.option('--backend-param', '-b', multiple=True)
def run_analyzedir(project_id, directory, suffix, force,
                   limit, threshold, backend_param):
    """"
    Analyze a directory with documents. Write the results in TSV files
    with the given suffix.

    USAGE: annif analyzedir <project_id> <directory> [--suffix=SUFFIX]
                            [--force=FORCE] [--limit=N] [--threshold=N]
    """
    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param)
    hit_filter = HitFilter(limit, threshold)

    for docfilename, dummy_subjectfn in annif.corpus.DocumentDirectory(
            directory, require_subjects=False):
        with open(docfilename) as docfile:
            text = docfile.read()
        subjectfilename = re.sub(r'\.txt$', suffix, docfilename)
        if os.path.exists(subjectfilename) and not force:
            click.echo(
                "Not overwriting {} (use --force to override)".format(
                    subjectfilename))
            continue
        with open(subjectfilename, 'w') as subjfile:
            for hit in hit_filter(project.analyze(text, backend_params)):
                line = "<{}>\t{}\t{}".format(hit.uri, hit.label, hit.score)
                click.echo(line, file=subjfile)


@cli.command('eval')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('subject_file', type=click.Path(dir_okay=False))
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
@click.option('--backend-param', '-b', multiple=True)
def run_eval(project_id, subject_file, limit, threshold, backend_param):
    """"
    Evaluate the analysis result for a document against a gold standard
    given in a subject file.

    USAGE: annif eval <project_id> <subject_file> [--limit=N]
           [--threshold=N] <document.txt
    """
    project = get_project(project_id)
    text = sys.stdin.read()
    backend_params = parse_backend_params(backend_param)
    hit_filter = HitFilter(limit=limit, threshold=threshold)
    hits = hit_filter(project.analyze(text, backend_params))
    with open(subject_file) as subjfile:
        gold_subjects = annif.corpus.SubjectSet(subjfile.read())

    template = "{0:<20}\t{1}"
    eval_batch = annif.eval.EvaluationBatch()
    eval_batch.evaluate(hits, gold_subjects)
    for metric, score in eval_batch.results().items():
        click.echo(template.format(metric + ":", score))


@cli.command('evaldir')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('directory', type=click.Path(file_okay=True))
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
@click.option('--backend-param', '-b', multiple=True)
def run_evaldir(project_id, directory, limit, threshold, backend_param):
    """"
    Evaluate the analysis results for a directory with documents against a
    gold standard given in subject files.

    USAGE: annif evaldir <project_id> <directory> [--limit=N]
           [--threshold=N]
    """
    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param)

    hit_filter = HitFilter(limit=limit, threshold=threshold)
    eval_batch = annif.eval.EvaluationBatch()
    for docfilename, subjectfilename in annif.corpus.DocumentDirectory(
            directory, require_subjects=True):
        with open(docfilename) as docfile:
            text = docfile.read()
        hits = hit_filter(project.analyze(text, backend_params))
        with open(subjectfilename) as subjfile:
            gold_subjects = annif.corpus.SubjectSet(subjfile.read())
        eval_batch.evaluate(hits, gold_subjects)

    template = "{0:<20}\t{1}"
    for metric, score in eval_batch.results().items():
        click.echo(template.format(metric + ":", score))


@cli.command('optimize')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('directory', type=click.Path(file_okay=False))
@click.option('--backend-param', '-b', multiple=True)
def run_optimize(project_id, directory, backend_param):
    """"
    Evaluate the analysis results for a directory with documents against a
    gold standard given in subject files. Test different limit/threshold
    values and report the precision, recall and F-measure of each combination
    of settings.

    USAGE: annif optimize <project_id> <directory>
    """
    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param)

    filter_batches = generate_filter_batches()

    for docfilename, subjectfilename in annif.corpus.DocumentDirectory(
            directory, require_subjects=True):
        with open(docfilename) as docfile:
            text = docfile.read()
        hits = project.analyze(text, backend_params)
        with open(subjectfilename) as subjfile:
            gold_subjects = annif.corpus.SubjectSet(subjfile.read())
        for hit_filter, batch in filter_batches.values():
            batch.evaluate(hit_filter(hits), gold_subjects)

    click.echo("\t".join(('Limit', 'Thresh.', 'Prec.', 'Rec.', 'F1')))

    best_scores = collections.defaultdict(float)
    best_params = {}

    template = "{:d}\t{:.02f}\t{:.04f}\t{:.04f}\t{:.04f}"
    for params, filter_batch in filter_batches.items():
        results = filter_batch[1].results()
        for metric, score in results.items():
            if score > best_scores[metric]:
                best_scores[metric] = score
                best_params[metric] = params
        click.echo(
            template.format(
                params[0],
                params[1],
                results['Precision (per document average)'],
                results['Recall (per document average)'],
                results['F1 score (per document average)']))

    click.echo()
    template2 = "Best {}:\t{:.04f}\tLimit: {:d}\tThreshold: {:.02f}"
    for metric in ('Precision (per document average)',
                   'Recall (per document average)',
                   'F1 score (per document average)',
                   'NDCG@5',
                   'NDCG@10'):
        click.echo(
            template2.format(
                metric,
                best_scores[metric],
                best_params[metric][0],
                best_params[metric][1]))


if __name__ == '__main__':
    cli()
