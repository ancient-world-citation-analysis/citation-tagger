This repository is a citation tagger tool built using Pytorch and Gensim's Word2Vec. What it does is very simple:

For every word in a citation entry, outputs whether it is part of Author(A), Date(D), Title(T), and Other(O).

This is part of a bigger project called [Ancient World Computational Analysis](https://digitalhumanities.berkeley.edu/ancient-world-computational-analysis-awca).

# Fast Usage Tips

## Single Citation Entry
```
>>> from CitationTagger import CitationTagger
>>> ct = CitationTagger()
>>> ct(['''Mariya Toneva and Leila Wehbe. 2019. Interpreting and improving naturallanguage
processing (in machines) with natural language-processing (in the
brain). arXiv preprint arXiv:1905.11833 (2019).'''])
--- Citation #1 ---
              Mariya  ->  A
              Toneva  ->  A
                 and  ->  T
               Leila  ->  A
              Wehbe.  ->  O
               2019.  ->  D
        Interpreting  ->  T
                 and  ->  T
           improving  ->  T
     naturallanguage  ->  T
          processing  ->  T
                 (in  ->  T
           machines)  ->  T
                with  ->  T
             natural  ->  T
 language-processing  ->  T
                 (in  ->  T
                 the  ->  T
             brain).  ->  T
               arXiv  ->  T
            preprint  ->  O
    arXiv:1905.11833  ->  T
             (2019).  ->  D
-------------------
```

## Multiple Citation Entry
```
>>> ct(['''Mariya Toneva and Leila Wehbe. 2019. Interpreting and improving naturallanguage
processing (in machines) with natural language-processing (in the
brain). arXiv preprint arXiv:1905.11833 (2019).''',
'''James W Tanaka and Martha J Farah. 1993. Parts and wholes in face recognition.
The Quarterly journal of experimental psychology 46, 2 (1993), 225–245.'''])
--- Citation #1 ---
              Mariya  ->  A
              Toneva  ->  A
                 and  ->  A
               Leila  ->  T
              Wehbe.  ->  A
               2019.  ->  A
        Interpreting  ->  A
                 and  ->  D
           improving  ->  T
     naturallanguage  ->  T
          processing  ->  T
                 (in  ->  T
           machines)  ->  T
                with  ->  T
             natural  ->  T
 language-processing  ->  T
                 (in  ->  O
                 the  ->  O
             brain).  ->  O
               arXiv  ->  O
            preprint  ->  O
    arXiv:1905.11833  ->  O
             (2019).  ->  D
-------------------
--- Citation #2 ---
               James  ->  A
                   W  ->  A
              Tanaka  ->  T
                 and  ->  A
              Martha  ->  O
                   J  ->  D
              Farah.  ->  T
               1993.  ->  T
               Parts  ->  T
                 and  ->  T
              wholes  ->  T
                  in  ->  T
                face  ->  T
        recognition.  ->  T
                 The  ->  T
           Quarterly  ->  T
             journal  ->  T
                  of  ->  T
        experimental  ->  T
          psychology  ->  T
                 46,  ->  T
                   2  ->  T
             (1993),  ->  D
            225–245.  ->  O
-------------------
```

# Citation Tags can be returned as a list
```
>>> tags = ct.tag_citations(['''Mariya Toneva and Leila Wehbe. 2019. Interpreting and improving naturallanguage
processing (in machines) with natural language-processing (in the
brain). arXiv preprint arXiv:1905.11833 (2019).''',
'''James W Tanaka and Martha J Farah. 1993. Parts and wholes in face recognition.
The Quarterly journal of experimental psychology 46, 2 (1993), 225–245.'''])
>>> print(tags)
[['A', 'A', 'T', 'T', 'A', 'A', 'A', 'D', 'O', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'O', 'O', 'O', 'O', 'O', 'D', 'O'], ['A', 'A', 'T', 'T', 'O', 'D', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'D', 'O']]
```