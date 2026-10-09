"""Compare four approved temporal feature sets using unchanged baseline settings."""
from improve import main

if __name__ == '__main__':
    main(candidates=[(name, 'sqrt') for name in
                     ['autocorr', 'second_diff', 'turning', 'temporal_all']],
         results_file='temporal_results.csv')
