# QTM Insole Recorder Kullanım Kılavuzu

## İçindekiler
1. [Giriş](#giriş)
2. [Sistem Gereksinimleri](#sistem-gereksinimleri)
3. [Başlarken](#başlarken)
4. [Arayüz Tanıtımı](#arayüz-tanıtımı)
5. [Yapılandırma](#yapılandırma)
6. [Veri Kaydı](#veri-kaydı)
7. [Gerçek Zamanlı Grafikler](#gerçek-zamanlı-grafikler)
8. [Sorun Giderme](#sorun-giderme)

## Giriş

QTM Insole Recorder, biyomekanik ve hareket analizi alanında çalışan araştırmacılar ve klinisyenler için tasarlanmış gelişmiş bir uygulamadır. Bu uygulama, Qualisys Track Manager (QTM) hareket yakalama sistemleri ile basınca duyarlı tabanlıklar arasında sorunsuz bir entegrasyon sağlayarak, eş zamanlı veri toplama ve gerçek zamanlı görselleştirme imkanı sunar. Bu kapsamlı çözüm, kullanıcıların hem hareket hem de basınç verilerini aynı anda yakalamasına, izlemesine ve analiz etmesine olanak tanır.

## Sistem Gereksinimleri

QTM Insole Recorder'ı yüklemeden önce, sisteminizin aşağıdaki gereksinimleri karşıladığından emin olun:

### Yazılım Bağımlılıkları:
- Python (PyQt6 arayüz çerçevesi)
- QTM RT SDK (qtm_rt paketi) - hareket yakalama entegrasyonu için
- PyQtGraph - gerçek zamanlı veri görselleştirmesi için

### Donanım Gereksinimleri:
- Aktif bir Qualisys Track Manager (QTM) sistemi
- Uyumlu basınç tabanlıkları
- Uygun senkronizasyon için sistemlerin aynı ağa bağlı olması gerekmektedir

## Başlarken

### Kurulum ve Başlatma:
Uygulamayı kullanmaya başlamak için, kurulum dizininizdeki `app.py` dosyasını çalıştırın. Başarılı bir şekilde başlatıldığında, ana uygulama penceresi karşınıza çıkacaktır. Bu pencere üç ana bölüme ayrılmıştır:

- **Sol panel**: Tüm yapılandırma seçeneklerini ve kontrol öğelerini barındırır. Kayıt parametrelerinizi ayarlamanıza ve veri toplama oturumlarınızı yönetmenize olanak tanır.
- **Sağ panel**: Verilerinizin gerçek zamanlı görselleştirmelerini gösterir, kayıt oturumlarınız sırasında anında geri bildirim sağlar.
- **Alt panel**: Sistem işleyişi ve önemli olaylar hakkında sizi bilgilendiren durum ve günlük mesajlarını içerir.

## Arayüz Tanıtımı

Uygulama arayüzü, tüm gerekli işlevlere sezgisel erişim sağlamak için temiz ve düzenli bir yerleşim korunarak dikkatlice tasarlanmıştır.

### Sol Panel Bileşenleri:

#### QTM Yapılandırması ve Kontrolleri
Bu bölüm, QTM sisteminizle bağlantı kurmanızı sağlar. Aşağıdakileri sağlamanız gerekecektir:
- "QTM Host" alanında QTM sunucusunun IP adresi
- Bağlantı için ilgili port numarası
- QTM sisteminiz kimlik doğrulama gerektiriyorsa şifre

"Connect" ve "Disconnect" düğmeleri, QTM bağlantısını gerektiği gibi yönetmenizi sağlar.

#### Tabanlık Yapılandırması ve Kontrolleri
Burada basınç tabanlık sisteminize bağlantıyı kurabilirsiniz. Yapılandırma şunları içerir:
- Tabanlık verilerinin alınacağı IP adresi
- Veri iletimi için UDP port numarası

Bağlantı durumu açıkça görüntülenir ve sağlanan düğmelerle bağlanabilir veya bağlantıyı kesebilirsiniz.

#### Kayıt ve Çıktı Ayarları
Bu bölüm, veri kayıt tercihlerinizi yönetir. Şunları yapabilirsiniz:
- Veri depolama için tercih ettiğiniz çıktı dizinini seçme
- Kayıt oturumlarını başlatma ve durdurma
- Mevcut kayıt durumunu izleme

#### Tabanlık Sütun Başlıkları
Bu özellik, tabanlık verilerinizin nasıl düzenleneceğini özelleştirmenize olanak tanır. Araştırma gereksinimlerinize uygun sütun başlıkları oluşturabilir ve gerektiğinde bunları ekleyebilir, düzenleyebilir, kaldırabilir ve yeniden sıralayabilirsiniz.

### Sağ Panel Özellikleri:
Görselleştirme paneli, çoklu sekmeler aracılığıyla gerçek zamanlı veri geri bildirimi sağlar. Her sekme, kayıt oturumunuzun farklı yönlerine adanmıştır. Bu tasarım, tüm kayıt oturumunuzun net ve düzenli bir görünümünü korurken belirli veri akışlarını izlemenize olanak tanır.

## Yapılandırma

Kayıt oturumunuzu ayarlamak, hem QTM hem de tabanlık sistemlerini yapılandırmayı içerir. Bu bölüm, süreci adım adım açıklar.

### QTM Sistemi Yapılandırması:
QTM sisteminizin IP adresini girerek başlayın. Varsayılan port genellikle 22222'dir, ancak kurulumunuza bağlı olarak değişebilir. Sisteminiz kimlik doğrulama gerektiriyorsa, belirlenen alana şifreyi girin. Tüm alanlar düzgün bir şekilde yapılandırıldıktan sonra, bağlantıyı kurmak için "Connect QTM" düğmesine tıklayın.

### Tabanlık Sistemi Yapılandırması:
Tabanlık sistemi için, verilerin alınacağı IP adresini belirtmeniz gerekecektir. Sistemi yerel olarak çalıştırıyorsanız, tüm mevcut arayüzlerden dinlemek için "0.0.0.0" veya "127.0.0.1" kullanabilirsiniz. Uygun UDP port numarasını girdikten sonra, veri almaya başlamak için "Connect Insole" düğmesine tıklayın.

### Çıktı Yapılandırması:
Kayıt oturumunuza başlamadan önce, veri çıktı tercihlerinizi ayarlamanız önemlidir. Kaydedilen veri dosyalarınız için bir dizin seçmek üzere "Browse" düğmesini kullanın. Ayrıca, sütun başlıklarını yapılandırarak tabanlık veri yapınızı özelleştirebilirsiniz. Bu, verilerinizi analiz ihtiyaçlarınıza en uygun şekilde düzenlemenize olanak tanır.

## Veri Kaydı

Kayıt süreci, veri bütünlüğünü ve uygun senkronizasyonu sağlarken basit olacak şekilde tasarlanmıştır.

### Kayıt İçin Hazırlık:
Bir kayıt oturumu başlatmadan önce, hem QTM hem de tabanlık sistemlerinin düzgün bir şekilde bağlı olduğunu ve çalıştığını doğrulayın. Arayüzdeki durum göstergeleri başarılı bağlantıları onaylayacaktır. Çıktı dizininizin ayarlandığından ve yeterli depolama alanının mevcut olduğundan emin olun.

### Kayıt Oturumu Başlatma:
Sistemleriniz hazır olduğunda, veri toplamaya başlamak için "Start Recording" düğmesine tıklayın. Uygulama otomatik olarak QTM ve tabanlık verileri için ayrı dosyalar oluşturacak ve iki veri akışı arasındaki zamansal hizalamayı sağlamak için yerleşik senkronizasyon kullanacaktır.

### Aktif Kayıt Sırasında:
Kayıt devam ederken, verilerinizi gerçek zamanlı görselleştirme sekmeleri aracılığıyla izleyebilirsiniz. Durum paneli, aktif veri toplamayı onaylamak için "Recording" gösterecek ve günlük paneli kayıt süreci hakkında sürekli durum güncellemeleri sağlayacaktır.

### Kayıt Oturumunu Sonlandırma:
Kaydı durdurmak için "Stop Recording" düğmesine tıklayın. Devam etmeden önce günlük panelindeki onay mesajını bekleyin. Veri dosyalarınız otomatik olarak belirttiğiniz çıktı dizinine kaydedilecek, düzgün bir şekilde biçimlendirilecek ve analiz için hazır olacaktır.

## Gerçek Zamanlı Grafikler

Görselleştirme sistemi, kayıt oturumunuzun çoklu veri akışları aracılığıyla kapsamlı gerçek zamanlı geri bildirim sağlar:

### Hareket Yakalama Görselleştirmesi:
Uygulama, QTM sisteminizden 3B işaretçi konumlarını göstererek, öznenin hareketini gerçek zamanlı olarak izlemenizi sağlar. Ek olarak, 6B (rijit cisim) verisi, segment yönelimleri ve konumları hakkında bilgi sağlar.

### Kuvvet ve Basınç Verileri:
Zemin tepki kuvvetlerini ve basınç merkezi hareketlerini göstermek için kuvvet plakası verileri görselleştirilir. Tabanlık basınç verisi ekranı, basınç dağılımı modelleri hakkında anında geri bildirim sağlar.

Bu veri türlerinin her biri kendi sekmesinde sunulur, böylece tüm veri akışlarına erişimi korurken kayıtınızın belirli yönlerine odaklanmanıza olanak tanır. Görselleştirme gerçek zamanlı olarak güncellenir ve veri toplama kaliteniz hakkında anında geri bildirim sağlar.

## Sorun Giderme

### Yaygın Sorunlar ve Çözümleri:

#### QTM Bağlantı Sorunları:
QTM sistemine bağlanmada zorluk yaşıyorsanız, ağ ayarlarınızı doğrulayın ve şunlardan emin olun:
- IP adresi ve port numarası doğru girilmiş
- QTM sistemi çalışıyor ve erişilebilir durumda
- Ağ bağlantınız stabil
- Güvenlik duvarı ayarları gerekli bağlantılara izin veriyor

#### Tabanlık Veri Alım Sorunları:
Tabanlık verileri düzgün alınmadığında, şunları kontrol edin:
- UDP port ayarları doğru ve diğer uygulamalar tarafından kullanılmıyor
- Tabanlık sistemi aktif olarak veri iletimi yapıyor
- Sistemler arasındaki ağ bağlantısı korunuyor

#### Kayıt Başlatma Sorunları:
Bir kayıt oturumu başlatırken sorunlarla karşılaşırsanız, şunları doğrulayın:
- Her iki sistem de düzgün bir şekilde bağlı ve hazır
- Seçilen çıktı dizininde uygun yazma izinleri var
- Kayıtınız için yeterli depolama alanı mevcut

#### Hata Mesajlarını Anlama:
Durum günlük paneli, sistem işleyişi ve oluşan hatalar hakkında ayrıntılı bilgi sağlar. Karşılaşabileceğiniz yaygın mesajlar şunları içerir:
- "Error: Failed to connect to QTM" - QTM bağlantısıyla ilgili bir sorunu gösterir
- "Error: QTM connection lost" - Kesintiye uğramış bir bağlantıyı bildirir
- "Could not parse insole data" - Tabanlık veri formatıyla ilgili sorunları gösterir
- "Error setting up insole listener" - Tabanlık bağlantısıyla ilgili sorunları önerir

Kalıcı sorunlarla karşılaştığınızda, her zaman ayrıntılı hata mesajları için günlük panelini kontrol edin ve tüm sistemlerin düzgün yapılandırıldığından ve çalıştığından emin olun. Sorunlar devam ederse, kayıt oturumunu yeniden başlatmayı denemeden önce tüm fiziksel bağlantıları ve sistem ayarlarını doğrulayın. 