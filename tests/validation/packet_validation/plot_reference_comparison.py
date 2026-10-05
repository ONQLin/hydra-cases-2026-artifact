#!/usr/bin/env python3
"""Plot CHIPSIM-referenced serving shifts and error/simulation-cost tradeoffs."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class BackendStyle:
    key: str
    label: str
    color: str
    marker: str
    linestyle: str


class ReferenceComparison:
    reference = 'chipsim_contended'
    styles = (
        BackendStyle('chipsim_contended', 'CHIPSIM (reference)', '#252525', 's', '-'),
        BackendStyle('hydra_packet', 'HYDRA-Packet', '#D55E00', '^', '--'),
        BackendStyle('hydra_sim', 'HYDRA-Analytic', '#0072B2', 'o', '--'),
    )
    metrics = (('tokens_per_sec', 'TP'), ('ttft_s', 'TTFT'))

    def __init__(self, csv_path, window):
        self.csv_path = csv_path
        self.window = window
        with csv_path.open() as stream:
            self.rows = list(csv.DictReader(stream))
        if not self.rows or len({r['point'] for r in self.rows}) != len(self.rows):
            raise ValueError('Require nonempty, uniquely labelled paired design points.')
        summary = json.loads(csv_path.with_name('summary.json').read_text())
        if summary['simulated_seconds'] != window or summary['paired_points'] != len(self.rows):
            raise ValueError('Window or coverage disagrees with the comparison summary.')
        if summary['missing_points'] or summary['paired_points'] != summary['expected_points']:
            raise ValueError('This figure requires a complete paired comparison.')
        for row in self.rows:
            for backend in self.styles:
                for key in ('tokens_per_sec', 'ttft_s', 'process_wall_seconds'):
                    column = backend.key + '_' + key
                    try:
                        value = float(row[column])
                    except (ValueError, KeyError) as error:
                        raise ValueError(f"{row['point']}: missing completed-run value {column}") from error
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"{row['point']}: {column} must be finite and positive.")
                    row[column] = value
        self.groups = sorted({(row['model'], row['dataset']) for row in self.rows})

    def group_rows(self, group):
        return sorted((r for r in self.rows if (r['model'], r['dataset']) == group),
                      key=lambda r: int(r['point'].rsplit('P', 1)[1]))

    def shift(self, row, backend, metric):
        return 100 * (row[backend + '_' + metric] / row[self.reference + '_' + metric] - 1)

    def statistics(self):
        reference_time = sum(r[self.reference + '_process_wall_seconds'] for r in self.rows)
        records = []
        for backend in self.styles:
            total = sum(r[backend.key + '_process_wall_seconds'] for r in self.rows)
            record = dict(backend=backend.key, label=backend.label, paired_points=len(self.rows),
                          process_wall_sum_s=total, speedup_vs_chipsim=reference_time/total,
                          time_cost_percent_of_chipsim=100*total/reference_time)
            for metric, _ in self.metrics:
                errors = [self.shift(r, backend.key, metric) for r in self.rows]
                record[metric + '_mape_percent'] = sum(abs(e) for e in errors)/len(errors)
                record[metric + '_mean_signed_shift_percent'] = sum(errors)/len(errors)
                record[metric + '_rmspe_percent'] = math.sqrt(sum(e*e for e in errors)/len(errors))
                record[metric + '_max_ape_percent'] = max(abs(e) for e in errors)
            records.append(record)
        return records


class ReferenceFigures:
    def __init__(self, comparison, output):
        self.comparison = comparison
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)

    def save(self, fig, name):
        import matplotlib.pyplot as plt
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(self.output / f'{name}.{extension}', dpi=220, facecolor='white')
        plt.close(fig)

    @staticmethod
    def format_axis(axis):
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(alpha=0.16, zorder=0)
        axis.tick_params(labelsize=10)

    def serving_shifts(self, include_shifts=True):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        data = self.comparison
        row_count = 2 if include_shifts else 1
        height = 8.6 if include_shifts else 4.9
        fig, axes = plt.subplots(row_count, len(data.groups), figsize=(6.2*len(data.groups), height), squeeze=False)
        for column, group in enumerate(data.groups):
            rows = data.group_rows(group)
            upper = axes[0, column]
            lower = axes[1, column] if include_shifts else None
            for style in data.styles:
                upper.plot([r[style.key+'_ttft_s'] for r in rows],
                           [r[style.key+'_tokens_per_sec'] for r in rows],
                           color=style.color, marker=style.marker, linestyle=style.linestyle,
                           linewidth=2 if style.key==data.reference else 1.6,
                           markersize=7, markerfacecolor=style.color if style.key==data.reference else 'white',
                           zorder=4 if style.key==data.reference else 3)
            for row in rows:
                reference = (row[data.reference+'_ttft_s'], row[data.reference+'_tokens_per_sec'])
                upper.annotate(row['point'].rsplit('-', 1)[1], reference,
                               xytext=(7, -13), textcoords='offset points', fontsize=10,
                               bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.6))
                for style in data.styles[1:]:
                    target = (row[style.key+'_ttft_s'], row[style.key+'_tokens_per_sec'])
                    upper.annotate('', xy=target, xytext=reference,
                                   arrowprops=dict(arrowstyle='->', color=style.color,
                                                   linewidth=1.1, alpha=.65, shrinkA=4, shrinkB=5))
                if lower is None:
                    continue
                endpoints = [(self.comparison.shift(row, style.key, 'ttft_s'),
                              self.comparison.shift(row, style.key, 'tokens_per_sec'))
                             for style in data.styles[1:]]
                lower.plot([p[0] for p in endpoints], [p[1] for p in endpoints],
                           color='#B0B0B0', linewidth=1, zorder=1)
                for style, endpoint in zip(data.styles[1:], endpoints):
                    lower.scatter(*endpoint, marker=style.marker, color=style.color,
                                  facecolors='white', s=62, linewidths=1.6, zorder=3)
                    # P1 shifts nearly coincide: label the pair once at its midpoint.
                    if row['point'].endswith('P1'):
                        continue
                    offset = (-8, -14) if style.key=='hydra_packet' else (7, 6)
                    lower.annotate(row['point'].rsplit('-', 1)[1], endpoint,
                                   xytext=offset, textcoords='offset points', fontsize=10,
                                   color=style.color, ha='right' if offset[0]<0 else 'left')
                if row['point'].endswith('P1'):
                    midpoint = tuple(sum(p[i] for p in endpoints)/2 for i in (0, 1))
                    lower.annotate('P1', midpoint, xytext=(-15, 9), textcoords='offset points', fontsize=10)
            upper.set(title=' / '.join(group), xlabel='Mean TTFT (s)', ylabel='Output TP (tokens/s)')
            upper.margins(x=.09, y=.14)
            self.format_axis(upper)
            if lower is not None:
                lower.axhline(0, color='#B0B0B0', linewidth=.8)
                lower.axvline(0, color='#B0B0B0', linewidth=.8)
                lower.scatter(0, 0, color=data.styles[0].color, marker='s', s=52, zorder=4)
                lower.annotate('CHIPSIM\n(0%, 0%)', (0, 0), xytext=(8, 6), textcoords='offset points', fontsize=9)
                lower.set(xlabel='TTFT shift relative to CHIPSIM (%)',
                          ylabel='TP shift relative to CHIPSIM (%)', title='Signed shift of the same design points')
                lower.margins(x=.25, y=.19)
                self.format_axis(lower)
        handles = [Line2D([], [], color=s.color, marker=s.marker, linestyle=s.linestyle,
                          markerfacecolor=s.color if s.key==data.reference else 'white', label=s.label)
                   for s in data.styles]
        title = 'Serving curves and shifts from CHIPSIM' if include_shifts else 'Serving curves with CHIPSIM as reference'
        fig.suptitle(f'{title} | {data.window:g} s window', fontsize=16, y=.99)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .96 if include_shifts else .92),
                   ncol=3, frameon=False)
        caption = ('Top: arrows point from CHIPSIM to the approximation. Bottom: gray lines pair the same design point.\n'
                   'Shift = 100 × (backend / CHIPSIM − 1). Curves connect selected points; the window starts at t = 0.')
        if not include_shifts:
            caption = ('Arrows point from CHIPSIM to the approximation at the same design point.\n'
                       'Curves connect selected points; the measurement window starts at t = 0. Compute and runtime are shared.')
        fig.text(.5, .025, caption, ha='center', fontsize=9, color='#4A4A4A')
        fig.subplots_adjust(left=.08, right=.96, bottom=.14 if include_shifts else .20,
                            top=.88 if include_shifts else .79, hspace=.49, wspace=.28)
        self.save(fig, 'reference_curves_and_shifts' if include_shifts else 'reference_curves')

    def accuracy_speedup(self):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        data = self.comparison
        records = {r['backend']: r for r in data.statistics()}
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
        for axis, (metric, title) in zip(axes, data.metrics):
            for style in data.styles:
                record = records[style.key]
                x, y = record['speedup_vs_chipsim'], record[metric+'_mape_percent']
                axis.scatter(x, y, color=style.color, marker=style.marker, s=100, zorder=4,
                             edgecolors='white', linewidths=.7)
                label = f'{style.label}\n{y:.2f}% MAPE · {x:,.1f}×'
                offset = (10, 12) if style.key==data.reference else (0, 13)
                if style.key=='hydra_sim':
                    offset = (-8, -34) if metric=='tokens_per_sec' else (-8, 13)
                axis.annotate(label, (x, y), xytext=offset, textcoords='offset points',
                              fontsize=10, color=style.color,
                              ha='right' if style.key=='hydra_sim' else ('left' if style.key==data.reference else 'center'))
            largest = max(r[metric+'_mape_percent'] for r in records.values())
            axis.set(xscale='log', xlim=(.7, 12000), ylim=(-max(.06, largest*.10), max(.2, largest*1.4)),
                     xlabel='Simulation speedup vs. CHIPSIM (log scale)', ylabel=f'{title} MAPE (%)',
                     title=f'{title} error vs. simulation speed')
            axis.set_xticks([1, 10, 100, 1000, 10000])
            axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}×'))
            axis.axhline(0, color='#A0A0A0', linewidth=.8)
            axis.text(.97, .045, 'Lower error ↓    Faster →', transform=axis.transAxes,
                      ha='right', fontsize=10, color='#555555')
            self.format_axis(axis)
        fig.suptitle(f'Accuracy–simulation cost tradeoff | {len(data.rows)} paired cases, {data.window:g} s each', fontsize=15, y=.98)
        fig.text(.5, .035, 'MAPE: equal weight per design point. Speedup: sum(CHIPSIM process wall time) / sum(backend process wall time).\n'
                 'CHIPSIM is the comparison reference; compute is shared. Wall times are single measurements under concurrent host load.',
                 ha='center', fontsize=9, color='#4A4A4A')
        fig.subplots_adjust(left=.07, right=.97, bottom=.24, top=.83, wspace=.26)
        self.save(fig, 'accuracy_vs_speedup')

    def write_statistics(self):
        data = self.comparison
        records = data.statistics()
        with (self.output / 'accuracy_cost_summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        summary = dict(reference=data.reference, simulated_seconds=data.window,
                       paired_points=len(data.rows), input_csv=str(data.csv_path),
                       input_csv_sha256=hashlib.sha256(data.csv_path.read_bytes()).hexdigest(),
                       mape_definition='100 * mean(abs(backend / CHIPSIM - 1)); equal weight per case',
                       rmspe_definition='100 * sqrt(mean((backend / CHIPSIM - 1)^2))',
                       speedup_definition='sum(CHIPSIM process wall seconds) / sum(backend process wall seconds)',
                       scope='Shared compute; transient from t=0; legacy TTFT excludes queue delay; summed wall time is not parallel makespan.',
                       results=records)
        (self.output / 'accuracy_cost_summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    def run(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                             'axes.titlesize': 12, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
            self.render()
        self.write_statistics()


    def render(self):
        self.serving_shifts()
        self.accuracy_speedup()


class DocumentationFigures(ReferenceFigures):
    """Curated serving curves, tradeoff plot and numeric table for documentation."""

    def summary_table(self):
        import matplotlib.pyplot as plt
        data = self.comparison
        records = data.statistics()
        cells = []
        for record in records:
            seconds = record['process_wall_sum_s']
            duration = f'{seconds/3600:,.2f} h' if seconds >= 3600 else f'{seconds:,.2f} s'
            cells.append([record['label'], f"{record['tokens_per_sec_mape_percent']:.2f}%",
                          f"{record['ttft_s_mape_percent']:.2f}%", duration,
                          f"{record['speedup_vs_chipsim']:,.2f}×"])
        fig, axis = plt.subplots(figsize=(12.4, 3.2))
        axis.set_axis_off()
        table = axis.table(cellText=cells,
                           colLabels=['Backend', 'TP MAPE', 'TTFT MAPE', 'Total process wall time', 'Speedup vs. CHIPSIM'],
                           colWidths=[.26, .14, .14, .23, .23], cellLoc='center', bbox=[0, .18, 1, .68])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor('#D8DEE5')
            cell.set_linewidth(.7)
            if row == 0:
                cell.set_facecolor('#EDF1F5')
                cell.get_text().set_weight('bold')
            else:
                cell.set_facecolor('white' if row % 2 else '#FAFBFC')
                if column == 0:
                    cell.get_text().set_color(data.styles[row-1].color)
                    cell.get_text().set_weight('bold')
        fig.suptitle(f'Multi-fidelity validation summary | {len(data.rows)} paired cases, {data.window:g} s each',
                     fontsize=15, y=.96)
        fig.text(.5, .06, 'CHIPSIM is the comparison reference. MAPE weights each design point equally.\n'
                 'Speedup uses summed process wall times; these sums are distinct from parallel elapsed time.',
                 ha='center', fontsize=9, color='#4A4A4A')
        fig.subplots_adjust(left=.025, right=.975, top=.86, bottom=.13)
        self.save(fig, 'validation_summary_table')

    def render(self):
        self.serving_shifts(include_shifts=False)
        self.accuracy_speedup()
        self.summary_table()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-csv', type=Path, required=True)
    parser.add_argument('--window', type=float, default=5)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figure-set', choices=['analysis', 'documentation'], default='analysis')
    args = parser.parse_args()
    figure_class = {'analysis': ReferenceFigures, 'documentation': DocumentationFigures}[args.figure_set]
    figure_class(ReferenceComparison(args.comparison_csv, args.window), args.output_dir).run()


if __name__ == '__main__':
    main()
