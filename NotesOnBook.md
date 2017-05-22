#Notes on Productivity and Reuse

##main problem:
- tradeoff between productivity and reuse
- what part do you store, what part to you compute each time
- if you want to store high frequency items, how do you infer a priori what will be high frequency and what is not 


##Chapt. 1
###models
- for these purposes, models are programs that generate observable data

###three approaches
- modularity: computation and storage at different levels of grammer (i.e. compute syntax, store phonology) 
- ideosyncrasy: identifies computation with regularity and storage with idiosyncracy (i.e. compute regular rule-based productions, store unique exceptions)
- modularity works on assumption that there are fewer morphemes than words, fewer words than sentences, etc. 
	- takes this as underlying principle for its grammar
- modularity defeated by ability to create entirely new words 
	- the more it relies on storage, the fewer novel forms it can generate, the more it has new forms, the less it can store
- idiosyncratic structures stored, all other structures are computed by rules and are productive
	- psycholinguistic evidence suggests this is not how we work
- **final approach: Learning Synergies**
	- different sorts of evidence make "similar, redundant predictions about the productivity of word-formation processes"

##the proposal
### 4 assumptions made
- producing new forms is based on system of recursive rules
- frequent sequences can be stored and reused as one unit
- combining sequences leads to inference problem
- this problem comes from fact that language learner wants to store fewer items **and** wants to minimize number of computational steps

### 4 possible strategies
- full parsing
	- store nothing, everything is computed from the smallest units 
	- Dirichlet-Multinomial Probabilistic Context-Free Grammars
- full listing
	- store everything you've seen in its entirety
	- Maximum a posteriori Adaptor Grammar
- Exemplar based
	- store all structures consistent with data 
	- store intermediate structure, but not whole thing
- Inference based
	- model that we will be using
	- can store abstract structures and specific ones, as well as intermediate
	- unlike first three, productivity and reuse determined by using inference 
		- for each data point, decide if it would be better to store it or to compute it later, or a little of both

##Chapt 2 
###what is inference
- inference is guessing hidden structure/generalizing from noisy/partial/unclear data
- represent data/hypotheses as random variables, use probabilistic conditioning
- usually have many random variables we're interested in
	- r.v.'s over "words, morphemes, parses, rules, and grammars"

##what is conditioning
- formula is Baye's rule
- 2 step procedure with two inputs
	- input 1: joint distribution
	- input 2: set of values for a subset of r.v. in joint distribution (conditioner)
	- conditioner can be assumptions, hypothesis, observed data, ..., but usually just observed data
	- procedure checks every comb. of values and discards if the conditioner doesn't hold (i.e. $P(D=d,l=l) = 0  \Rightarrow P(L=l | D = d) = \frac{P(D=d, L=l)}{\sum\limits_{\forall l' \in L}^{} P(D = d, L = l')} = 0$
	- after discarding inconsistent ones, procedure takes probability mass of rejected outcomes and redistributes over remaining possibilities proportionally (this is denominator $\sum\limits_{\forall l' \in L}^{} P(D = d, L = l')$ 

##rationality
- we are assuming that cognitive systems exist to help find optimal solutions
	- not always evolutionarily true, but allows us to look for optimal solution
- we can do rational analysis: specify goals of system, formalize computational space to reach the goals, derive optimal behavior of system trying to reach goals

##Stochastic memoization
- problem: if we memoized even in PPL, we will end up with determinate output after one iteration
- we need stochastic memoization
	- associates input with distribution over output values 
	- probability that each previously computed value will be reused in next call
	- try to infer: which computations likely to have been stored and reused in generated data

###Pitman-Yor Process
- to do this we need PYP
- non-parametric (can change given new evidence)
- restaurant with infinite number of tables, 1st customer sits at first table, every customer after that sits at table $i$ with probability $\frac{y_i - a}{N + b}$ where $N$ is number of customers, $y_i$ is number of people at table $i$, and $a,b$ are discount and concentration parameters, and sits and a new table with probability $\frac{Ka + b}{N + b}$ where $K$ is total number of occupied tables
- if we use PYP as memoization, each table is a reusable computation sampled from the memoized function. 
	- **modeling English morphology:**
	-  tables are word types
	-  customers are word tokens
- PYP has simplicity bias 
	- higher probability if restaurant has:
	-  fewer customers
	-  fewer tables
	-  for fixed N, assigns customers to fewest tables
- rich get richer scheme
	- prefers most popular tables
	- smaller, simpler lexica
- **encodes competing bias as well:**
	- probability of a new table has direct relationship with $K$, number of occupied tables
	- goes up the more tables are occupied
	- so probability of generating novel form goes up the more novel words you've seen
	
###lazy evaluation
- could either eval things as they are called
- or could wait until argument value needed before computing it
	- this can allow memoization if arg value has already been computed before, avoids unnecessary computations
- delayed objects are promises (promise to return a value when needed)
	- "forced" when later returned
- lazy eval can change function of probabilistic programs
	- lazy can still save us time though
	- so we should try and learn when to use lazy, when to use eager
- can apply this stochastic laziness to unpacking RHS of grammar
	- randomly pick whether to unpack RHS symbol or to delay sampling till later
- now we need a way of storying partial computations

####Fragment Grammars
- model results of stochasitcally memoizing unfold procedure
- unfolding fragment grammar involves either
	- returning previously sampled memoized partial computation  
	- or sample new partial subcomputation from unfold procedure
- promise computations are ultimately forced, no incomplete expressions
- since grammar is recursive, computation may be memoized, saving computations

####Fragment $\lambda$
- can be used to define procedures
- its procedules automatically reuse partial subcomputations (stochastically)
	- if the arguments of the function are delayed, then delay the body
	- otherwise execute the body
- can be used for many types of models 
	
###overview of models thus far
####Dirichlet-Multinomial PCFG (DMPCFG)
- full parsing (only smallest units stored, no larger trees/subtrees reused)
- rule probabilities come from a Dirichlet prior dist. 
- posterior distribution comes from data, we can then calculate conditional
- DMPCFG weights are related to frequency in training data
	- high posterior rules usually proportional to frequency in data

####Maximum A Posteriori Adaptor Grammar (MAG)
- full listing (everything stored in entirety)
- use grammar with max score under posterior dist. of training data
- stores all computations, but only reuses them probabilistically
	- will be reused proportionally to frequency in subtree
	- prefers to reuse computed subtree, but could form a new one
- how much reuse depends on $a$ and $b$ hyperparameters to PY distribution (discount and concentration)
	- they can be inferred from the data
- original PCFG rule estimation proportional to number of stored computations in which they are used, so based on type-frequency of rule in stored trees
- type-frequency can be good estimator of productivity
	- so MAG can be good for some domains
- but not always the case, so MAG limited
	- can't learn a productive combination of morphemes like -ability

####Data Oriented Parsing (DOP)
- exemplar based
- store **all** subtrees
- all possible generalizations in data are stored 
- many techniques are DOP
	- differ mostly in probability estimation of stored fragments
- DOP1
	- probability of subtree proportional to token frequency in training data
	- problems:
		- biased and inconsistent
		- overweights training data nodes appearing higher up in tree/in larger trees, so bias for larger stored items
- Equal-node Estimator (ENDOP)
	- assigns equal weight to training data nodes and each data item
	- improved performance for syntax compared to DOP1
- mostly stores all subtrees
	- some penalize larger trees or long derivations
- ENDOP and DOP1 store all subtrees

####Fragment Grammars (FG)
- inference-based
- tries to find balance between storage and productivity to best predict training data
- inherits from MAG, can store any sub-computation
- generalizes MAG to allow sub-computations to include variables needing computation
	- MAG can only compute forms using PCFG, FG can reused stored structures with variables (lazy evaluate variables?) 
	- also estimates prob. of storage based on token frequency and prob of PCFG rules based on type frequency
- not possible to efficiently compute max a posteriori FG directly from training data, uses stochastic search to find most probable grammar
- commit to one analysis for each form in data
- difference from other models:
	- other models have a fixed policy for storage
	- same thing will always be either stored or reused 
	- FG: stored subtrees for given form depend on previously seen data
	- chooses optimal shared structure between new tree and previous ones
- result of this structure:
	- wants all instances of individual word types to be parsed in same way
	- same with particular sub-forms

	
##Chapt 3: math details
###CFGs
- what is CFG
	- Context-Free Grammar
	- 4-tuple: $G = \<V_G, T_G, R_G, W  \>$ 
	- $V_G$: non-terminal symbols (finite set)
	- $T_G$: terminal symbols (finite set)
	- $R_G \subseteq V_G x (V_G \cup T_G)*$: production rules (finite set)
	- $W \in V_G$: unique start symbol
- non-terminals usually capital letters
- in syntax they would be constituent categories (XP's and X-bars)
- start symbol usuall S for sentence
- W used instead, since focusing on words
	- terminals would be morphemes/atomic words
- rules are $A \rightarrow \gamma$ where $\gamma$ is sequence of terminals and non-terminals, $A$ is non-terminal
- lhs(rule), rhs(rule) return left or right hand side of rule
- language $L_A$ associated with non-terminal $A$ is set of expressions derivable from that nonterminal through recursive application of production rules
	- language of whole CFG G is $L_W$ 
- no policy for how to choose rules

###Multinomial Context Free Grammars
- multinomial dist. is easiest way to specify dist. over finite number of discrete choices
	- if you have $K$ possible choices, specify a vector with length $K$ $\Theta$ such that $\Theta_i \geq 0$ and $\sum\limits_{\forall i} \Theta_i = 1$
	- probability of rules given by multinomial distribution
- derivation/parse tree represents the computation that was taken 
	- tree is complete if all leaves are terminals
	- fragment if it has leaves that are nonterminals
- function yield(d)
	- returns leaves of derivation d as list
- function root(d)
	- returns root of derivation d
- function top(d)
	- returns the top-most production rule (i.e. depth = 1) 
- nonterminal $A$ derives expression $w$ if $\exists$ complete derivation such that root(d) = $A$ and yield(d) = $w$. 
- formal: multinomial PCFG $\<G, \{\Theta^A\}_{A\in V_G}$\>  is CFG $G$ with set of probability vectors $\{\Theta^A\}_{A\in V_G}$ where each vector represents the parameters of multinomial distribution over set of rules that share $A$ on left hand side. 
- write $\Theta_{r}^{A}$ or $\Theta_r$ to mean component of vector $\Theta^A$ associated with rule $r$. 
- then $\sum\limits_{r \in R^A} \Theta_{r}^{A} = 1$
- formal equation states that probability of tree d given by $G$ is product of probability of rules used to build tree from depth-one subtrees, summed over all rules whose left hand side matches root $d$ and whose right hand side matches immediate children of the root $d$. Usually for PCFGs there is only one such rule, so the summation is only over one term. 
- PCFGs make strong independence assumptions
	- expanding nonterminal happens without outside information
	- expressions are generated independently from one another
	- so given multinomial PCFG $G$, probability of set of computations can be computed directly from counts of how rules used in set without knowing where the rules were used
	- probability of a derivation $d$ is just product of probabilities of rules it contains
	- probability of expression is summing (marginalizing) over all derivation trees that share expression as their yield
- **inside probability** of sequence of terminals $w$ given rule $r$ is probability that the sequence is the yield of some complete tree for $w$ whose topmost depth-one subtree corresponds to $r$. 
- let **corpus of derivation trees** D of size $N_D$ be trees resulting from deriving $N_D$ expressions from start symbol $W$ 
- let $X = \{x^A\}$ be set of count vectors for each rule used in D
- function counts
	- takes set of derivation trees, returns corresponding vectors of rule counts
	- counts(D) = X 
- then the probability of a corpus of derivations given by product of probabilities of all the rules used in the corpus

###Dirichlet-multinomial distribution
- Polya-urn representation
- places Dirichlet prior on vector $\Theta$ of probabilities, parametrizing multinomial distribution over $K$ elements 
- 2 step sampling process
	- draw prob. vector $\Theta$ from Dirichlet
	- draw $N$ observations from multinomial characterized by $\Theta$ 
- another way of thinking about it:
	- let pseudo-counts be counts of imaginary prior observations of each of $K$ possible outcomes in multinomial 
	- 1st observation is sampled with probability proportional to pseudocounts
	- the N+1th observation sampled proportionally to  (number of previous draws of that outcome) + (pseudo-count associated with that outcome)
	- **question** more about pseudo-count? is that the original Dirichlet prior for the multinomial?
	- increases probability of each outcome as more of that outcome sampled 
	- like Chinese restaurant but instead of tables, bins of multinomial dist. 
	- **draws are not independent** but they are exchangeable 
- Dirichlet multinomial basically assigns probabilities to different partionings of N objects into K bins
	- so different possibilities for multinomial distributions
- probability of a certain partition:
	- $P(x | \pi ) = \frac{\prod_{i=1}^K \Gamma(\pi_i + x_i) \Gamma(\sum_{i=1}^K \pi_i)}{\Gamma(\sum_{i=1}^k \pi_i + x_i) \prod_{}^ \Gamma(\pi_i)}$
	- same as integrating out $\Theta$ from product of multinomial likelihood with Dirichlet prior
###application to PCFG
- normal multinomial PCFG has $\Theta$ pre-specified 
- but $\Theta$ could be drawn from Dirichlet prior
	- this is DMPCFG
	- Multinomial-Dirichlet Probabilistic Context Free Grammar
- DMPCFG is a full parsing approach
- uses Polya-urn process for each nonterminal in CFG
	- each nonterminal gets a vector of pseudocounts 
###DOP
- tree substitution grammar 
	- generalization of CFG
	- basic units can be arbitrary tree fragments, rather than rules
	- uses substitution where nonterminal leaf node in tree fragment replaced with label in nonterminals where another tree fragment has that label as its root. 
	- CFG is special case of TSG where all fragments are restricted to be depth-one trees
- probabilistic TSG has each subtree with probability
- prefix(d) enumerates prefixes of derivation tree d
- equation for probability of derviation is essentially same as recursive stochastic eqn for PCFGS except that prefixes are used instead of rules
- **come back to this **

###Pitman-Yor
- reduces to single-parameter Chinese restaurant process when a=0. 
- on average, $a$ is limiting proportion of tables in the restaurant with only one customer, and limiting probability of sitting at a new table
- $b$ controls the rate of growth of new tables in relation to total number of customers
- each table has dish $v$ which is the label on the table shared by all customers sitting at it
	- when customer picks a new table, dish is sampled from separate distribution $\mu$ which is the **base distribution** and is now the dish being served at that table
- like Dirichlet-multinomial dist., PY is a distribution over ways of partitioning N customers to K tables. 
	- different from Dirichlet-multinomial dist., number of partitions $K$ is unbounded 
	- probability of a particular partition is product of probability of $N$ choices made in seating $N$ customers 
	- order does not matter, so dist is exchangeable 
- let $mem{G}$ be vector of probabilities. 
	- $mem{G}$ ~ PYP(a,b,G)$ 
	- unlike $\Theta ~ DIRICHLET(\pi)$ $mem{G}$ is countably infinite 
	- so we have to do lazy enumeration

###Adaptor Grammars
- can be understood as PYP-memoization of PCFG unfold procedure
- MAGs (Max a posteriori Adaptor Grammars) adds Dirichlet priors to rule weights in CFG system 
- always returns fully expanded tree
	- so AG can only store/reuse complete tree fragments
- also exchangeable
- probability of particular set of alayses given in terms of counts associated with individual Dirichlet-multinom. and PY dists. 

###Fragment Grammars
- can store partial derivations in PYP memoizer
	- lazy evaluation   
- decision to recurse or halt at each non-terminal not made by fair coin flip
	- beta-binomial distribution, biased coin
	- can be inferred during training
	- learns separate prob. of recursing or delaying for each nonterminal in each RHS for each rule
	- beta dist. is analog of Dirichlet with only 2 outcomes
- fragment grammar made of CFG, $\{\pi^A\}A\in V_g $ which are vectors of Dirichlet-multinomial pseudocounts for each nonterminal, set of PY hyperparameters for each noterminal, and set of pseudo-counts for the beta-binomial distribution associated with RHS of each production rule in CFG

####FG inference
- trying to answer: what is dist. over sets of stored tree fragments that best explains observed data
- given gramman F_1 and correct parse trees D, try to find dist. over set of fragment grammar analyses for parses which specifies ways parses can be split up into tree fragments
- find $P(F|D, F_1) = P(D|F_1, F) P(F|F_1)$

###Metropolis-Hastings sampler
####MCMC
- Markov chain Monte-Carlo
- approximate inference technique 
- samples from hypothesis space by defining Markov chain of local steps
- here, hypothesis consists of set of all possible fragment grammar analyses $F$ given some input
- needs a transition kernel $k:F \rightarrow F'$ which defines distribution over transitions from one hypothesis to another by making small changes to current hypothesis
- define transition kernel whose long-term behavior converges to desired target posterior dist. $P(F|D,F_1)$
- kernel uses exchangeability of fragment grammar
	- can treat any word in corpus as if it were most recently observed item
- steps:
	- choose target item from training set $d^{(i)}$
	- remove tree fragments and other structures associated with its current analysis $f^{(i)}$ from current state
	- call this new state $F_{-f^{(i)}}$
	- sample new analysis $f^{(i)}'$ conditioned on $F_{-f^{(i)}}$, $F_1$ hyperparameters, and parse tree $d^{(i)}$
	- either accepted or rejected by Metropolis-Hastings criterion 
	- add either $f^{(i)}$ or $f^{(i)}'$ depending on if proposal was accepted or rejected to state of system and procedure is repeated for next one

###PCFG approximation
- PCFG has strong conditional independence assumption
	- means that you can have efficient algorithm to solve PCFG parsing problems
	- rely on fact that distributions over parses for a string generated by PCFG decomposes into product over independent sums of substring parses
- decomposition does not work for case of Polya-urn representation of Dirichlet mutlinomial PCFGs, PYAGs, or PYFGs.
	- Prob of using tree/rule fragment not independent of other tree/rule fragments
	- because of rich get richer dynamic
- so inference algorithm samples proposal analyses by making use of PCFG approximation of true FG dist. but which has independence assumption.
	- let that be "approximating PCFG"
	- make rule for each stored fragment in each PY restaurant, assign it prob. proporitional to prob. of table
	- add rules from Dirich.-multinom. PCFG base system, with weights according to table prob. for restaurant of LHS of rule.
- with approximating grammar, we can efficiently compute dist. over derivations of an expression  
- see book for 3 steps on how to compute proposal for Metropolis-Hastings sampler 

###speeding everything up
- can use batch initialization for MAP fragment grammar
	- every node in input corpus assigned in parallel to own table in restaurant
	- all decisions about dist. of other r.v.'s randomly sampled from prior
	- starts the sampler at random, initially has low probability
	- then increases prob by merging tables that have same label
- can use type/token binning
	- can take tokens many at a time by grouping according to type 
	- resample whole bins of tokens 
	- best to run at multiple bin sizes
- can condition on constituent information
	- condition sampler on gold standard parses 
- auxiliary variable slice sampling
	- parsing is most time costly operation
	- main cause is large size of approximating PCFGs
 	- so adapt Metropolis-Hastings to be faster

 
 
##Questions:
- what exactly are you sampling  

	
	
	