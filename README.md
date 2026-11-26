# simonph-ml
Så i första runnen (full_run.png) Hade jag ett enkelt neuralt nätverk med en som var 28*28 -> 128, Relu, 128 -> 10. Med mse loss. Efter den runnen implmenterade jag early stopping och körde extra många epocher. 
Sen visade det sig att den aldrig stoppade early så det blev en run med bara med epocher.
Den första fick en accuracy av ~95% den andra fick en accuracy av 97.3%. 
Jag bytte sen loss från mse till crossentropyloss. Detta gjorde så att modelen fick en accuracy av 97.6% och triggrade early stopping (precis).