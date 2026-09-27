# NLP21 + Jigsaw + CTC با نویز گاوسی

این بسته روی کد NLP21 ضمیمه‌شده ساخته شده است. در مسیر جدید هیچ اتک، PGD،
FGSM، InfoNCE یا گرادیان‌گیری برای ساخت نویز وجود ندارد. فایل‌های trainer قدیمی
صرفاً برای حفظ پروژه‌ی اولیه داخل بسته باقی مانده‌اند؛ کامندهای این راهنما وارد مسیر جدید می‌شوند.

```python
noisy = clean + noise_std * torch.randn_like(clean)
```

`noise_std` انحراف معیار نویز است، نه واریانس. مقدار صفر نویز را خاموش می‌کند.
نویز فقط هنگام train و فقط در binهای معتبر نمونه‌گیری می‌شود. padding و ارزیابی CER نویز ندارند.

## چهار مدل انتخاب‌شده

| arm | window / gap | lossها با وزن‌های اصلی | head |
| --- | --- | --- | --- |
| `proposed` | 10 / 1…8 | reconstruct + position + 0.5 × pair | DeepSets |
| `order_lag` | 10 / 1…8 | reconstruct + position + 0.5 × pair + lag | DeepSets، lag غیرخطی با ۶ کلاس |
| `order_time_axis` | 10 / 1…8 | reconstruct + lag | lag خطی با ۸ کلاس |
| `order_full` | 6 / 3…5 | reconstruct + position + 0.5 × pair + lag | DeepSets، lag با ۸ کلاس، hard mining، reset head هر ۴۰۰ update |

این‌ها چهار تنظیم مدل/هدف هستند؛ همگی از trunk residual کد استاد استفاده می‌کنند.
همه reconstruction دارند. reconstruction مانند کد اصلی، cross entropy روی فعالیت
quantize‌شده‌ی هر نورون و هر bin است، نه MSE. forecast خاموش است.

برای خواسته‌ی «فقط نویز گاوسی»، neuron dropout، gain jitter و circular roll کد استاد
پیش‌فرض خاموش‌اند. بنابراین `order_full` پیش‌فرض، نسخه‌ی سازگار با آزمایش Gaussian-only
است، نه بازتولید دقیق همه‌ی augmentationهای ران Perich. برای فعال‌کردن تنظیمات اصلی
این augmentationها، `--source_augment` را اضافه کن؛ همان فایل `experiment.json` تنظیم مؤثر را ذخیره می‌کند.
dropout داخل GRU همان `--dropout 0.4` کامند خودت باقی مانده است.

مدل `order_gapped` هم قابل انتخاب است. کنترل‌های `reconstruct_only`، `anchor_default`
و `anchor_full` موجودند، ولی کامند پیش‌فرض آن‌ها را اجرا نمی‌کند. فایل `mobile_jigsaw.py`
در ضمیمه‌ها موجود نبود؛ این بسته ادعای پشتیبانی از MobileNetهای آن فایل ندارد.

## چهار نوع آموزش برای هر مدل

| train_mode | noise_site | مرحله اول | مرحله دوم / مسیر CTC |
| --- | --- | --- | --- |
| `frozen` | `neural` | SSL روی داده‌ی نورونی + Gaussian؛ target reconstruction تمیز | encoder و headها freeze؛ decoder با CTC روی embedding تمیز |
| `frozen` | `embedding` | SSL روی داده‌ی تمیز | encoder و headها freeze؛ Gaussian روی embedding؛ فقط decoder با CTC |
| `joint` | `neural` | یک مرحله‌ی مشترک | داده‌ی نورونی نویزی به هر دو شاخه‌ی SSL و CTC می‌رود؛ target reconstruction تمیز |
| `joint` | `embedding` | یک مرحله‌ی مشترک | SSL روی داده‌ی تمیز؛ Gaussian روی embedding ورودی decoder؛ گرادیان CTC به encoder هم می‌رسد |

در حالت joint، `loss = CTC + ssl_weight × SSL` و `ssl_weight=1` است.
در frozen هیچ CTCای در مرحله‌ی اول محاسبه نمی‌شود و هیچ SSLای در مرحله‌ی دوم آموزش داده نمی‌شود.
فریز شامل پارامترهای encoder و تمام headهای SSL است؛ headهای SSL در inference لازم نیستند.
**منظور از frozen/neural در این بسته، نویز در pretrain است؛ مرحله‌ی decoder آن تمیز است.**
نویز embedding به خود اسکالر loss اضافه نمی‌شود؛ CTC از پیش‌بینیِ حاصل از embedding نویزی محاسبه می‌شود.

## اتصال به NLP21

```text
normalized neurons [B,T,192]
  -> Gaussian smoothing (sigma=2, kernel=20؛ مثل داده‌ی اولیه)
  -> Jigsaw encoder, natural windows W=10 یا W=6
  -> embeddings [B,T,64]
  -> unfolding(kernel=32, stride=4)
  -> packed bidirectional GRU(hidden=1024, layers=5, dropout=0.4)
  -> 32 logits (31 کاراکتر + blank)
  -> CTC
```

نویز `neural` بعد از normalization و قبل از smoothing اضافه می‌شود.
نویز `embedding` بعد از encoder و قبل از unfolding اضافه می‌شود.
encoder در مسیر SSL و CTC مشترک است. مسیر متوالی به‌صورت fully convolutional
پیاده شده و در تست، با اجرای encoder اصلی روی تک‌تک پنجره‌ها برابر است.
برای هر trial، مرزها جداگانه pad می‌شوند و هیچ span جیگساو از مرز trial عبور نمی‌کند.

این اتصال عمداً Jigsaw را روی ۱۹۲ کانال واقعی اعمال می‌کند. در کد CEBRA اولیه،
encoder می‌توانست بعد از unfolding باشد. بنابراین ورودی GRU جدید 64×32 است؛
این کار صرفاً تعویض یک لایه‌ی هم‌شکل با CEBRA نیست و receptive field آن هم یکسان نیست.
`--kernel 32` مربوط به پنجره‌بندی decoder است، نه window داخلی Jigsaw.
`--offset` و `--random_offset` مربوط به نمونه‌گیری contrastive قبلی‌اند؛ در این مسیر
اثر ندارند و اگر داده شوند پیام واضح چاپ می‌شود.

## نصب و اجرای همه‌ی حالت‌ها

داخل پوشه‌ی استخراج‌شده‌ی `nlpich-main`، همان محیطی را فعال کن که torch با CUDA دارد.
این مسیر به نصب CEBRA یا `edit_distance` احتیاج ندارد. وابستگی‌هایش torch، numpy و scipy هستند.
برای نصب وابستگی‌های غایب: `python -m pip install -r requirements-jigsaw.txt`.
torch سازگار با CUDA را از محیط فعلی خودت نگه دار؛ اجرای واقعی روی GPU اینجا تست نشده است.

کامند زیر ۴ مدل × ۴ حالت = **۱۶ train و ۱۶ CER** را به‌ترتیب اجرا می‌کند:

```bash
python -u run_jigsaw_ctc_matrix.py --datasetPath /data/hossein/mm_project/CORP_data_release --out_root nlp21-jigsaw-gauss08 --noise_std 0.8 --pretrain_steps 6000 --nBatch 20000 --seed 0
```

این اسکریپت تنظیمات GRU کامند خودت را می‌گذارد: `--gru --gauss_in --bidir --batchSize 16 --hidden 1024 --dropout 0.4 --layers 5 --kernel 32 --stride 4`.
تمام کامندهای واقعی را هم قبل از اجرا چاپ می‌کند.

**۶۰۰۰ یعنی optimizer update در pretrain، نه ۶۰۰۰ epoch ران Perich.**
در frozen، مجموعاً ۶۰۰۰ update برای SSL + ۲۰۰۰۰ update برای CTC داریم.
در joint، ۲۰۰۰۰ update مشترک داریم. این دو از نظر تعداد کل update برابر نیستند؛
اگر مقایسه با بودجه‌ی برابر می‌خواهی، تعدادها را صریحاً تنظیم کن.

ضریب را مثلاً به `--noise_std 0.2` تغییر بده و یک `out_root` جدید انتخاب کن.
عدد یکسان در فضای نورونی و embedding الزاماً شدت نسبی یکسانی ندارد؛ embedding
عمداً مطابق encoder استاد L2-normalize نمی‌شود.

برای دیدن کامندها بدون شروع train، `--dry_run` را اضافه کن.
برای ادامه‌ی اجراهای متوقف‌شده همان کامند را با `--resume` بزن؛ حالت‌های تمام‌شده
از checkpoint پایان‌یافته عبور می‌کنند و بقیه ادامه می‌یابند. تنظیمات مدل/بودجه در resume باید یکسان باشند.
checkpoint شامل optimizer، شماره update، ترتیب/cursor داده و وضعیت RNGهاست.
برای ارزیابی مجدد همه‌ی مدل‌ها، همان کامند را با `--run eval` بزن.

## اجرای یک مدل و یک حالت

```bash
python -u start_trainer.py --jigsaw --datasetPath /data/hossein/mm_project/CORP_data_release --out_dir nlp21-proposed-frozen-neural --jigsaw_arm proposed --train_mode frozen --noise_site neural --noise_std 0.8 --pretrain_steps 6000 --nBatch 20000 --gru --gauss_in --bidir --batchSize 16 --hidden 1024 --dropout 0.4 --layers 5 --kernel 32 --stride 4 --seed 0
```

برای سه حالت دیگر فقط `--train_mode` و `--noise_site` را طبق جدول عوض کن و
یک `out_dir` جدا بده. نام arm نیز یکی از چهار نام جدول باشد.
در کامندهای جدید `--adv`، `--adv_eps`، `--adv_norm`، `--no_noise` و
`--constantOffsetSD` را استفاده نکن؛ parser جدید آن‌ها را قبول نمی‌کند.

ارزیابی:

```bash
python -u eval_single_model.py --datasetPath /data/hossein/mm_project/CORP_data_release --out_dir nlp21-proposed-frozen-neural
```

`eval_single_model.py` از `config.json` نوع مدل را تشخیص می‌دهد. entrypointهای
مستقیم `train_jigsaw_ctc.py` و `eval_jigsaw_ctc.py` نیز موجودند.

برای فقط `order_lag` با هر چهار حالت:

```bash
python -u run_jigsaw_ctc_matrix.py --datasetPath /data/hossein/mm_project/CORP_data_release --out_root nlp21-lag-gauss08 --arms order_lag --noise_std 0.8 --pretrain_steps 6000 --nBatch 20000 --seed 0
```

## اجرای RunAI

اسکریپت `submit_nlp21_jigsaw.sh` همان image، PVC و منابع job قبلی را استفاده می‌کند.
به آن مسیر **واقعی پوشه‌ی کد در PVC home** را بده. مثلاً فقط اگر کد را در مسیر زیر
استخراج کرده‌ای:

```bash
bash submit_nlp21_jigsaw.sh /home/mirzaei/nlpich-main mm8-nlp21-jigsaw 0.8
```

بخش داخل `bash -lc` یک خط است و آرگومان‌های پایتون با آرایه ساخته می‌شوند؛
مشکل جداشدن `--epochs` یا اسم مدل‌ها از دستور پایتون تکرار نمی‌شود.
job از اینجا submit نشده است. قبل از مصرف GPU می‌توانی matrix را با `--dry_run` ببینی.

## داده، CER و خروجی‌ها

- train: `seed_model_training_data/mat` به‌جز آخرین block هر فایل، مطابق loader اولیه.
- validation حین train: آخرین block هر فایل seed؛ داده‌های online برای آموزش یا انتخاب بهترین checkpoint استفاده نمی‌شوند.
- eval پیش‌فرض `--split all`: seed heldout + هر دو پوشه‌ی online، همان مجموعه‌ی eval کد ارسالی. ترتیب تجمیع trialها برای logits جدید با فایل قدیمی یکسان نیست.
- `--split heldout` و `--split online` هم موجودند. CER برابر مجموع edit distance تقسیم بر مجموع کاراکترهای هدف است، نه میانگین CERهای trial.
- خروجی پیش‌فرض از آخرین checkpoint و بدون نویز/augmentation است. `--checkpoint best` بهترین CER روی heldout را انتخاب می‌کند؛ اگر آن را انتخاب کردی برای گزارش مستقل `--split online` مناسب‌تر است، چون heldout برای انتخاب استفاده شده است.
- normalization همان `get_input` اولیه است: آمار هر block از همان block محاسبه می‌شود، از جمله blockهای ارزیابی. این preprocessing از داده‌ی بدون برچسبِ ارزیابی استفاده می‌کند و نباید به‌عنوان normalization کاملاً train-only معرفی شود.
- quantileهای reconstruction فقط از binهای train، با نمونه‌گیری تصادفی تا `--quantile_samples 100000`، محاسبه و در checkpoint ذخیره می‌شوند. از padding، target نویزی یا داده‌ی eval استفاده نمی‌شود.

در `out_root` فایل‌های `cer_summary.csv` و `cer_table.md` ساخته می‌شوند.
جدول برای هر arm چهار ستون CER دارد: frozen/neural، frozen/embedding، joint/neural، joint/embedding.
هر مدل پوشه‌ی جداگانه با `config.json`، `experiment.json`، `checkpoint.pt`،
`modelWeights`، `modelWeights.best`، `training.jsonl` و `eval_all_last.json` دارد.
در frozen، `pretrained.pt` هم قبل از آموزش decoder ذخیره می‌شود.

`--save_logits` در eval، logits با ترتیب کاراکترهای همین مدل را در فایل `.pt` ذخیره می‌کند؛
reordering مخصوص language model در تابع `fix_logits` قدیمی به‌صورت ضمنی انجام نمی‌شود.

## تست‌ها و تفاوت‌های اجرایی مهم

```bash
python -m unittest discover -s tests -v
```

تست CPU شامل ۱۶ ترکیب، تغییر واقعی پارامترها، فریز، gradient از CTC به encoder،
Gaussian با sigma مشخص، padding، هم‌ارزی encoder پنجره‌ای و متوالی، clean target،
CTC تکرار کاراکتر، reset head و ادامه‌ی دقیق update بعدی از checkpoint است.
یک تست CLI روی فایل‌های MAT مصنوعی نیز train، ذخیره، eval و جدول چهارستونه را بررسی کرده است.
**هیچ CER واقعی CORP یا ادعای بهبود عملکرد با این بسته گزارش نشده است.**

طول خروجی unfolding به شکل `floor((T-kernel)/stride)+1` محاسبه می‌شود.
GRU با طول‌های واقعی pack می‌شود تا padding در جهت backward اثر نگذارد.
CTC با float32 و بررسی alignment محاسبه می‌شود؛ نمونه‌ی غیرقابل‌تراز silently حذف نمی‌شود.
مستند API: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html

optimizer: Adam برای هر دو گروه، SSL با lr=0.001 و eps=1e-8؛ decoder با
lr خطی 0.02→0.002 و eps=0.1 مطابق تنظیمات پایه‌ی NLP21. برخلاف trainer قبلی،
در هر update یک backward/step داریم. reset head در `order_full` به ازای updateهای
SSL شمارش می‌شود و اینجا stateهای Adam آن head نیز پاک می‌شوند.

primitives معماری از فایل‌های استاد بدون تغییر استخراج شده‌اند؛ مشخصات و SHA256 منابع
در `nlp21_jigsaw/SOURCES.json` ثبت شده است. منطق training برای trialهای متغیرطول و CTC جدید است.
