# AI_devs 3, Lekcja 1, Moduł 1 — Interakcja z dużym modelem językowym

![Cover Image](https://cloud.overment.com/S02E03-1731372201.png)

W dzisiejszych czasach generatywne AI zyskuje na popularności nie tylko w kontekście przetwarzania tekstu i audio, ale także w dziedzinie manipulacji obrazem i generowania nowych grafik. Chociaż projektowanie graficzne nie było tradycyjnie związane z programowaniem, narzędzia takie jak [Midjourney](tools/Midjourney.md) i [Stable Diffusion](glossary/Stable%20Diffusion.md) stają się istotnym elementem tej relacji, oferując nowe możliwości nie tylko dla projektantów, ale także dla deweloperów tworzących rozwiązania oparte na AI.

## Narzędzia AI w grafice

Przykładem praktycznego zastosowania generatywnego AI jest [picthing](https://pic.ping.gg/), narzędzie stworzone przez Theo — t3.gg, które efektywnie usuwa tło ze zdjęć, gwarantując poprawę jakości tego procesu. Inny przykład to Pieter Levels i jego projekty [PhotoAI](https://photoai.com/) oraz [InteriorAI](https://interiorai.com), które demonstrują, jak AI może rozwiązywać konkretne problemy w dziedzinie edycji zdjęć i projektowania wnętrz.

![Picthing Example](https://cloud.overment.com/2024-09-26/aidevs3_picthing-a5ce0e6a-b.png)
![Pieter Levels Projects](https://cloud.overment.com/2024-09-26/aidevs3_levelsio-55df5a6e-5.png)

## Możliwości i ograniczenia generowania obrazu

Dokonano dużych postępów w generowaniu obrazów przez AI, jak pokazują porównania z ostatnich dwóch lat ([Comparing AI-generated images two years apart — 2022 vs. 2024](https://medium.com/@junehao/comparing-ai-generated-images-two-years-apart-2022-vs-2024-6c3c4670b905)). Choć jakość obrazów znacznie się poprawiła, nadal występują ograniczenia w generowaniu złożonych elementów, takich jak tekst czy szczegóły rąk.

![Progress in Image Generation](https://cloud.overment.com/2024-09-26/aidevs3_midjourney-79dd9b18-9.png)
![Image Generation Limitations](https://cloud.overment.com/2024-09-26/aidevs3_peace-955577e3-1.png)

Platformy takie jak [Replicate](tools/Replicate.md) i [Leonardo.ai](https://leonardo.ai/) oferują dostęp do modeli AI przez API, co jest szczególnie interesujące z punktu widzenia deweloperów. Alternatywnie, [RunPod](https://blog.runpod.io/how-to-get-stable-diffusion-set-up-with-comfyui-on-runpod/) umożliwia hosting modeli z dostępem do GPU, co może być przydatne w przypadku bardziej zaawansowanych aplikacji.

## Generowanie grafik na potrzeby marketingowe

Generowanie grafik oparte na szablonach jest niezwykle przydatne w kontekście marketingu, gdzie potrzebne są różne formaty tej samej kreacji. Przykładem mogą być szablony okładek wydarzeń na eduweb.pl, które można szybko dostosować do nowych treści z minimalnym wysiłkiem.

![Templates Example](https://cloud.overment.com/2024-09-26/aidevs3_eduweb-b678b9a8-5.png)

## Techniki projektowania promptów

W generatywnej grafice, podobnie jak w przypadku modeli LLM, kluczowe znaczenie ma tworzenie odpowiednich promptów. Przykłady pokazują, że zarówno szczegółowe opisy sceny, jak i prostsze, oparte na słowach kluczowych, mogą prowadzić do interesujących wyników.

![Detailed Midjourney Prompt](https://cloud.overment.com/2024-09-26/aidevs3_mj-852d34e2-4.png)
![Keyword-focused Prompt](https://cloud.overment.com/2024-09-26/aidevs3_mj2-f541fdfa-d.png)

Strategia tworzenia promptów polega na eksperymentowaniu oraz łączeniu różnych podejść, co często przynosi najlepsze rezultaty.

## Iteracyjne projektowanie stylu

Wprowadzenie koncepcji "meta promtpów" ułatwia utrzymanie spójnego stylu w generowanych grafikach. Ilustruje to przykład generowania awatarów, gdzie zachowana została jednolita stylistyka poprzez użycie jednego spójnego promptu z opcją modyfikacji szczegółów postaci.

![Consistent Style in Avatars](https://cloud.overment.com/2024-09-27/aidevs3_avatars-939ec7b0-3.png)

## Hosting modeli i API

Komercyjne zastosowania generatywnej grafiki coraz częściej wymagają programowalnego dostępu do modeli. platformy takie jak Replicate czy Leonardo.ai oferują efektywne usługi w tym zakresie, ułatwiając integrację AI z istniejącymi aplikacjami.

## Przyszłość generatywnej grafiki 

Narzędzia takie jak [ComfyUI](ComfyUI) czy [HTMLCSStoImage](https://htmlcsstoimage.com) rozwijają swoje możliwości w programowaniu oraz automatyzacji kreatywnych procesów. Umożliwiają one dynamiczne generowanie grafik przy użyciu szablonów HTML i stylów CSS, co upraszcza, np. tworzenie grafik na potrzeby marketingowe.

![HTMLCSStoImage Process](https://cloud.overment.com/2024-09-27/aidevs3_htmlcsstoimage-c6f590af-a.png)

## Podsumowanie i możliwości rozwoju

Generatywna grafika otwiera nowe ścieżki zarówno w automatyzacji, jak i innowacji w obszarach tradycyjnie zależących od ręcznej pracy graficznej. Przykłady produktów takich jak [ComfyUI](ComfyUI) pokazują, że nawet bez znaczących zasobów, można tworzyć wartościowe i efektywne rozwiązania dla marketingu i produkcji. Warto poświęcić czas na eksplorację dostępnych narzędzi, które nie tylko wzbogacają nasz warsztat, ale także mogą stać się kluczem do nowych możliwości zawodowych.

**WAŻNE:** Jeżeli twój sprzęt nie jest wystarczająco wydajny do pracy z ComfyUI, a nie chcesz inwestować w płatne narzędzia, możesz pomijać poniższy film.

<div style="padding:75% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/1029104946?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="02_03_comfy"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

Powodzenia!