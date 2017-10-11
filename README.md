# nlp-project-one

Handy links
------------

http://www.katrinerk.com/courses/python-worksheets/language-models-in-python

http://nlpforhackers.io/language-models/

https://stackoverflow.com/questions/37504391/train-ngrammodel-in-python

https://www.inf.ed.ac.uk/teaching/courses/fnlp/Tutorials/1_LMs/lab1.pdf

The data
--------

You will again process the novels and short stories at <https://sites.google.com/site/comp498004/data>. Take the last two digits of your student number (choose either of your team members' numbers).

- 00 through 15: Dickens's A Tale of Two Cities, Barnaby Rudge, Bleak House, David Copperfield, Dombey and Son, Great Expectations.
- 16 through 25: Dickens's Hard Times, Life and Adventures of Martin Chuzzlewit, Little Dorrit, Oliver Twist, The Life and Adventures of Nicholas Nickleby, The Old Curiosity Shop.
- 26 through 70: George Eliot's works.
- 71 through 99: the works of Elizabeth Gaskell.

Tokenizers
----------

Before I tell you what to do, a word of advice: avoid NLTK's tokenizer. I wrote a very primitive tokenizer which is quite competitive in terms of quality. (You can easily filter out all digits if you so wish. That may be wrong for general texts, but for Victorian novels it is just the ticket.) The NLTK tokenizer is also incredibly slow, a bit random, and often too smart for its own good.

I ran both tokenizers on the combined text of twelve novels by Dickens. The NLTK method takes over 40 seconds on my Mac; the silly tokenizer needs a mere two seconds. I compared their operation on "A Tale of Two Cities": the first 995 tokens are identical. (My elementary code misfires a little on hyphenated words over a line break.) On "Sense and Sensibility": the first 1096 tokens. (My banal code beats the NLTK tool on not attaching single quotes to following letters, and on not rewriting double quotations marks in LaTeX style. To be fair, it loses big way on not recognizing the Saxon genitive, a.k.a. the possessive 's.)

To sum up, use my simplistic tokenizer. Call trivialTokenizer(text) instead of nltk.word_tokenize(text).

https://sites.google.com/site/comp498004/code/testingTrivialTokenizer.py

Well, or write a better tokenizer and let me have a copy.

The task
--------

At last, we come to the theme of this project. It is language modelling, and in particular building a bigram model (in the style of the Berkeley restaurant project discussed the class notes). Note first of all that NLTK's conditional frequency distribution is not it. It only accounts for bigrams in the text, not for all possible bigrams over the vocabulary.

It is computationally much too costly to build a good model of a large literary text. That is because such a text can have a vocabulary of well over 15,000 word types. One would need a truly large and fast machine to build a model with over 225,000,000 probabilities.

We will construct an approximation. Get the training text. Tokenize it. Ask NLTK for a plain frequency distribution of these token, and then for V most frequent types. The choice of V is rather essential. I have successfully tried 8000, but the process becomes sluggish. With maybe 4000, it is not so bad.

Such an approximation is not a truly accurate model, but with larger values of V you will only shed part of the hapax or dis legomena, not a great loss. One could also make a provision for unknown words, but I will not ask for that.

I am sure I need not tell you that it makes sense to develop your Python code on smallish texts and lowish values of V, and only then run the actual test on a substantial data and with a satisfyingly large V.

The actual model will sit in a square matrix indexed in rows and columns by those V words. There will be V**2 slots initialized to zeroes, most of them expected to stays zeroes. NLTK will give you the bigram list. I very strongly suggest cleaning it up by keeping only the bigrams with both words among the top V; otherwise you may get strange results.

The count is best done in a nested dictionary of V items. Each item has a word type as a key; the associated value is a dictionary of V items. The values in those inner items are frequency counts. Dictionaries offer fast access to cells. Counting in a list of lists would be insanely expensive.

On the other hand, lists are more easily navigated, and can be sorted. If you feel you need a list of lists, you can construct it at a reasonable cost after the model has been built in a dictionary. In Python, a one-liner (!) converts a nested dictionary into a nested list, sorted by keys on top and by probabilities at the second level. Or you can turn a two-level dictionary into a one-level dictionary with a sorted list as the value.

By the way, after you have built a model and either of the list versions, you may want to save both structures for later use, and so avoid rerunning the model construction. Get acquainted with the Python module called pickle.

[Why not use arrays, you may ask. "A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers." That is Python's best array equivalent -- and entirely useless in this project.]

Once you have counted all bigrams -- and all unigrams, whose count is necessary to compute the probabilities -- you will perform Laplace smoothing. Adding 1 to each count is a bad, bad idea. You want to add a small number, but not an arbitrary number. My suggestion is this. Let L be the count of low-frequency bigrams, maybe those with 1-9 occurrences; increase each bigram count by 1/L, and each unigram count by V/L.

This brings us to the last step. You will turn smoothed counts into Maximum Likelihood Estimate probabilities, the simplest kind possible: bigram count over unigram count (as in the class notes). Once that has been done, I suggest a sanity check. The probabilities in every row of the table should add up to 1, or perhaps something like 0.99999... or 1.00000... since a perfect 1.0 is quite unlikely.

The tIming
----------

Let me comment on the processing time. (The largest part of the cost would be the conversion into a nested list, if you wanted it, and "pickling" that list. I did not time this part for large vocabularies, but it would run for some minutes if not longer.) For a large value of V, each traversal of the nested dictionary is a matter of several to around a dozen seconds. I ran a fairly extreme test on Dickens's twelve novels and a vocabulary of 8000 in around 120 seconds; it was 75 seconds for 6000, and 30 seconds for 4000. Text size is not much of a factor: Austen's "Sense and Sensibility" -- 25 times smaller -- and 8000 word types took over 80 seconds.

The tests
---------

The bigram model has been built. You ought to show me now, in whatever way you see fit, that this model performs the assigned task.

You also need to use those probabilities. Generate some new text from your model as in the Shannon game. Try to favour more frequent bigrams over those very very infrequent former zeroes. The current word keys into the posible next words; select more often the next words with the highest probabilities.

Here is another test you will run. Take two sentences composed of words in your model's vocabulary. Calculate which of them is more probable.

Bonus
-----

There is a wide variety of other things you can do when you have a language model. Let me list a few as hints. You can try any of them for a small bonus.

- Determine whether men or women are mentioned more often. Lists of male and female given names are not hard to find. Such names ought to appear in bigrams of relatively high probability. Also to consider: the titles Mr.,  Mrs. and Miss.

- Determine the main characters' profession or occupation. Wikipedia helpfully list professions and occupations. Again, each relevant terms should be part of a highly probable bigram.

- Build a model each for two long novels in your set, and identify really frequent bigrams typical of one but not of the other.

- Model two novels again, and determine which one uses more highly probable emotion words.

There would be a fair bit of bonus if you applied Good-Turing or Witten-Bell  smoothing. You would have to research the details on your own.

Finally, a bonus which I do not expect you to take up, but cannot help noting: build a trigram model. (Seriously, do not try it. It would be much harder than it seems, mainly because the cost would be prohibitive.)

The submission
--------------

Hand in a ZIP archive with your table-building program, your test (or tests) in Python, your bonus findings in a text file, and a brief note about how to run all this code. Do not send me the novels! Include a fairly detailed documentation if you prefer not to present your work in person.

Name your ZIP archive YourFamilyName_YourStudentNumber_project_1.zip (and email to me with a .txt extension added, just in case).
