

## scripts

```
$ scripts/download_from_aihub.sh -d ./raw_data/korean_dialect -k 71517 -p ~/aihubshell -a ...
$ scripts/unzip.sh -d raw_data -q -r
```

```
$ (balaenoptera) jinmang2@DESKTOP-029MHGN:~/ko_dialect$ python -c "
import cProfile, pstats
from scripts.prepare_data import _process_single_json
p = 'raw_data/korean_dialect/139-1.중·노년층_한국어_방언_데이터_(강원도,_경상도)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_02._경상도_03._2인발화/talk_set1_collectorgs51_speakergs208_speakergs209_12_0_215.json'
cProfile.run('_process_single_json(p)', 'prof.out')
pstats.Stats('prof.out').sort_stats('cumulative').print_stats(15)
"
Thu Jun  4 09:30:03 2026    prof.out

         51057 function calls (37585 primitive calls) in 0.084 seconds

   Ordered by: cumulative time
   List reduced from 95 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.084    0.084 {built-in method builtins.exec}
        1    0.000    0.000    0.084    0.084 <string>:1(<module>)
        1    0.002    0.002    0.084    0.084 /home/jinmang2/ko_dialect/scripts/prepare_data.py:116(_process_single_json)
       20    0.000    0.000    0.054    0.003 /home/jinmang2/ko_dialect/scripts/prepare_data.py:78(summarize_intonation)
 13502/40    0.046    0.000    0.052    0.001 {built-in method builtins.sum}
    13482    0.006    0.000    0.051    0.000 /home/jinmang2/ko_dialect/scripts/prepare_data.py:87(<genexpr>)
        1    0.021    0.021    0.022    0.022 {method 'read' of '_io.TextIOWrapper' objects}
        1    0.003    0.003    0.003    0.003 {orjson.loads}
       20    0.001    0.000    0.001    0.000 /home/jinmang2/ko_dialect/scripts/prepare_data.py:80(<listcomp>)
18058/18055    0.001    0.000    0.001    0.000 {built-in method builtins.len}
        1    0.000    0.000    0.001    0.001 <frozen codecs>:319(decode)
        1    0.001    0.001    0.001    0.001 {built-in method _codecs.utf_8_decode}
        1    0.000    0.000    0.001    0.001 /home/jinmang2/miniconda3/envs/balaenoptera/lib/python3.11/re/__init__.py:225(compile)
        1    0.000    0.000    0.001    0.001 /home/jinmang2/miniconda3/envs/balaenoptera/lib/python3.11/re/__init__.py:272(_compile)
        1    0.000    0.000    0.001    0.001 /home/jinmang2/miniconda3/envs/balaenoptera/lib/python3.11/re/_compiler.py:740(compile)

(balaenoptera) jinmang2@DESKTOP-029MHGN:~/ko_dialect$ # 무거운 talk_ 파일 — 운율 수정 후 새 프로파일
python -c "
import cProfile, pstats
from scripts.prepare_data import _process_single_json
p = 'raw_data/korean_dialect/139-1.중·노년층_한국어_방언_데이터_(강원도,_경상도)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_02._경상도_03._2인발화/talk_set1_collectorgs51_speakergs208_speakergs209_12_0_215.json'
cProfile.run('_process_single_json(p)', 'prof.out')
pstats.Stats('prof.out').sort_stats('tottime').print_stats(10)
"
Thu Jun  4 10:34:08 2026    prof.out

         24113 function calls (24103 primitive calls) in 0.017 seconds

   Ordered by: internal time
   List reduced from 95 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.004    0.004    0.005    0.005 {method 'read' of '_io.TextIOWrapper' objects}
        1    0.003    0.003    0.003    0.003 {orjson.loads}
    13482    0.002    0.000    0.002    0.000 /home/jinmang2/ko_dialect/scripts/prepare_data.py:86(<genexpr>)
        1    0.002    0.002    0.017    0.017 /home/jinmang2/ko_dialect/scripts/prepare_data.py:117(_process_single_json)
       40    0.001    0.000    0.003    0.000 {built-in method builtins.sum}
        1    0.001    0.001    0.001    0.001 {built-in method _codecs.utf_8_decode}
       20    0.001    0.000    0.001    0.000 /home/jinmang2/ko_dialect/scripts/prepare_data.py:80(<listcomp>)
       20    0.001    0.000    0.001    0.000 /home/jinmang2/ko_dialect/scripts/prepare_data.py:189(<listcomp>)
        1    0.000    0.000    0.000    0.000 /home/jinmang2/miniconda3/envs/balaenoptera/lib/python3.11/re/_compiler.py:243(_optimize_charset)
      448    0.000    0.000    0.000    0.000 /home/jinmang2/ko_dialect/scripts/prepare_data.py:72(t2s)

(balaenoptera) jinmang2@DESKTOP-029MHGN:~/ko_dialect$ python -c "
import time, glob
from scripts.prepare_data import _process_single_json
files = glob.glob('raw_data/korean_dialect/**/talk_*.json', recursive=True)[:200]
t = time.perf_counter()
n = sum(len(_process_single_json(f)) for f in files)
dt = time.perf_counter() - t
print(f'{len(files)} files, {n} samples, {dt:.2f}s, {len(files)/dt:.0f} files/s/core, {n/dt:.0f} samples/s/core')
"
200 files, 2233 samples, 0.46s, 434 files/s/core, 4844 samples/s/core
(balaenoptera) jinmang2@DESKTOP-029MHGN:~/ko_dialect$ 
```