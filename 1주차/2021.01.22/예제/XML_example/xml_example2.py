from bs4 import BeautifulSoup

with open("US08621662-20140107.XML", "r", encoding="utf8") as xml_file:
    patent = xml_file.read()

soup = BeautifulSoup(patent, "lxml")

publication_reference_tag = soup.find("publication-reference")
p_reference = publication_reference_tag.get_text()
# p_document_id_tag = publication_reference_tag.find("document-id")
# p_country = p_document_id_tag.find("country").get_text()
# p_doc_number = p_document_id_tag.find("doc-number").get_text()
# p_kind = p_document_id_tag.find("kind")
# p_date = p_document_id_tag.find("date")

application_reference = soup.find("application-reference")
a_reference = application_reference.get_text()  
# a_document_id_tag = application_reference.find("document-id")
# a_country = a_document_id_tag.find("country").get_text()
# a_doc_number = a_document_id_tag.find("doc-number").get_text()
# a_date = a_document_id_tag.find("date").get_text()

p_reference_list = p_reference.replace("\n", ' ').split()
a_reference_list = a_reference.replace("\n", ' ').split()
print(p_reference_list)