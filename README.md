# simonph-ml
Så i första runnen (full_run.png) Hade jag ett enkelt neuralt nätverk med en som var 28*28 -> 128, Relu, 128 -> 10. Med mse loss. Efter den runnen implmenterade jag early stopping och körde extra många epocher. 
Sen visade det sig att den aldrig stoppade early så det blev en run med bara med epocher.
Den första fick en accuracy av ~95% den andra fick en accuracy av 97.3%. 
Jag bytte sen loss från mse till crossentropyloss. Detta gjorde så att modelen fick en accuracy av 97.6% och triggrade early stopping (precis).

Efter det så testade jag köra en simple lr schedular där jag helt enkelt multiplicerade med 0.9 efter varje epoch.
Det gav lite bättre resultat men beror nog främst på lite random.

Jag fokuserade sen på att göra den lite snabbare. Hade fått reda på två grejer. Den första är att inte räkna ut deriv på sista lagret då det inte användes. Detta är också det största lagret by far så man sparar en del. Jag hade också fått höra att iterators skulle vara mycket snabbare än for loopar vilket värkade stämma... En annan grej jag hittade är att pga floats inte uppfyller egenskapen (a+b) +c = a+(b+c) så optimerar inte compilern det. Man kan istället använda rusts algebraic_add som får compilern att optimera utifrån det beteendet. Lite skummt att skriva det i den här ordningen men den sista grejen var mest key.

Efter det så testade jag att flattena weights men det gav inte så mycket (i alla fall jämfört med det andra).
När jag körde modelen så gav den en testscore på 98% vilket är ett pb. Vet inte hur jag alltid lyckas få bättre resultat när det basically är random. 